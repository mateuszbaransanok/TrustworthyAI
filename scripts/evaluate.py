import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from time import sleep
from typing import Any

from lightning_lite import seed_everything
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from trustworthyai.data.modules.datamodule import BaseDataModule
from trustworthyai.data.modules.transformer_datamodule import TransformerDataModule
from trustworthyai.data.modules.vision_datamodule import VisionDataModule
from trustworthyai.postprocessors.postprocessor import Postprocessor
from trustworthyai.postprocessors.transformer.dice import TransformerDICE
from trustworthyai.postprocessors.transformer.energy import TransformerEnergy
from trustworthyai.postprocessors.transformer.gradnorm import TransformerGradNorm
from trustworthyai.postprocessors.transformer.kl_matching import TransformerKLMatching
from trustworthyai.postprocessors.transformer.knn import TransformerKNN
from trustworthyai.postprocessors.transformer.msp import TransformerMSP
from trustworthyai.postprocessors.transformer.react import TransformerReAct
from trustworthyai.postprocessors.transformer.vim import TransformerVIM
from trustworthyai.postprocessors.vision.dice import VisionDICE
from trustworthyai.postprocessors.vision.energy import VisionEnergy
from trustworthyai.postprocessors.vision.gradnorm import VisionGradNorm
from trustworthyai.postprocessors.vision.kl_matching import VisionKLMatching
from trustworthyai.postprocessors.vision.knn import VisionKNN
from trustworthyai.postprocessors.vision.msp import VisionMSP
from trustworthyai.postprocessors.vision.odin import VisionODIN
from trustworthyai.postprocessors.vision.react import VisionReAct
from trustworthyai.postprocessors.vision.vim import VisionVIM
from trustworthyai.settings import EXPERIMENTS_DIR
from trustworthyai.training.classifier import BaseClassifier
from trustworthyai.training.transformer_classifier import TransformerClassifier
from trustworthyai.training.vision_classifier import VisionClassifier
from trustworthyai.utils.checkpoints import get_checkpoint_epoch, get_checkpoint_path
from trustworthyai.utils.config import get_experiment_name, load_experiments

LOGGER = logging.getLogger()

DATA_MODULES: dict[str, type[BaseDataModule]] = {
    'vision': VisionDataModule,
    'transformer': TransformerDataModule,
}

MODELS: dict[str, type[BaseClassifier]] = {
    'vision': VisionClassifier,
    'transformer': TransformerClassifier,
}

METHODS: dict[str, dict[str, type[Postprocessor]]] = {
    'vision': {
        'msp': VisionMSP,
        'energy': VisionEnergy,
        'odin': VisionODIN,
        'gradnorm': VisionGradNorm,
        'knn': VisionKNN,
        'vim': VisionVIM,
        'react': VisionReAct,
        'dice': VisionDICE,
        'kl_matching': VisionKLMatching,
    },
    'transformer': {
        'msp': TransformerMSP,
        'energy': TransformerEnergy,
        'gradnorm': TransformerGradNorm,
        'knn': TransformerKNN,
        'vim': TransformerVIM,
        'react': TransformerReAct,
        'dice': TransformerDICE,
        'kl_matching': TransformerKLMatching,
    },
}


@dataclass(kw_only=True)
class TrainerConfig:
    gpu: list[int] | None


@dataclass(kw_only=True)
class EvaluationConfig:
    seed: int
    task_name: str
    training_name: str
    evaluation_name: str
    datasets: list[str]
    method_name: str
    method_config: dict[str, Any]
    trainer: TrainerConfig


def main(config: EvaluationConfig, save_dir: Path) -> None:
    # SETUP ----------------------------------------------------------------------------------------
    seed_everything(config.seed)

    training_dir = EXPERIMENTS_DIR / 'training' / config.task_name / config.training_name
    ckpt_path = get_checkpoint_path(
        directory=training_dir.joinpath('checkpoints'),
        mode='best',
    )

    # DATAMODULE -----------------------------------------------------------------------------------
    datamodule = DATA_MODULES[config.task_name].load_from_checkpoint(str(ckpt_path))
    datamodule.prepare_data()
    datamodule.setup()  # type: ignore[call-arg]

    # MODEL ----------------------------------------------------------------------------------------
    model = MODELS[config.task_name].load_from_checkpoint(str(ckpt_path))

    # METHOD ---------------------------------------------------------------------------------------
    method = METHODS[config.task_name][config.method_name](
        model=model,
        **config.method_config,  # type: ignore
    )

    # TRAINER --------------------------------------------------------------------------------------
    logger = WandbLogger(
        project='TrustworthyAI',
        name=save_dir.name,
        version=save_dir.name,
        group=save_dir.name,
        job_type='evaluation',
        save_dir=str(save_dir),
        resume='allow',
    )
    logger.log_hyperparams(asdict(config))

    # TRAINER --------------------------------------------------------------------------------------
    trainer = Trainer(
        accelerator='gpu' if bool(config.trainer.gpu) else 'cpu',
        devices=config.trainer.gpu,
        logger=logger,
        enable_checkpointing=False,
        inference_mode=False,
    )

    try:
        # CONFIGURE METHOD -------------------------------------------------------------------------
        method.to('cuda' if bool(config.trainer.gpu) else 'cpu')
        method.configure(datamodule)

        # EVALUATE IN-DISTRIBUTION DATA ------------------------------------------------------------
        LOGGER.info("Evaluating in-distribution dataset")
        outputs = trainer.predict(method, dataloaders=datamodule.test_dataloader())
        method.setup_postprocessor(outputs)  # type: ignore[arg-type]

        # EVALUATE OUT-OF-DISTRIBUTION DATA --------------------------------------------------------
        logger.log_metrics({'epoch': get_checkpoint_epoch(ckpt_path)})
        all_metrics = []
        for dataset in config.datasets:
            LOGGER.info("Evaluating out-of-distribution dataset: %s", dataset)
            dataloader = method.create_dataloader(datamodule, dataset)
            metrics = trainer.test(method, dataloaders=dataloader)
            all_metrics.extend(metrics)
    except Exception as exc:
        LOGGER.error("Experiment error", exc_info=exc)
        sleep(5)
    else:
        # SAVING METRICS ---------------------------------------------------------------------------
        with save_dir.joinpath('metrics.json').open('w') as file:
            json.dump(
                obj=all_metrics,
                fp=file,
                indent=2,
                ensure_ascii=False,
                allow_nan=True,
            )
    finally:
        logger.experiment.finish()


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s]\t%(threadName)10s\t%(name)-20s\t%(levelname)10s\t%(message)s',
    )

    experiments = [
        (filename, experiment)
        for filename in [
            'params/evaluation/vision-cifar10.yaml',
            'params/evaluation/transformer-news-category.yaml',
            'params/evaluation/transformer-reviews.yaml',
            'params/evaluation/transformer-news-groups.yaml',
        ]
        for experiment in load_experiments(EvaluationConfig, filename)
    ]

    for exp_num, (exp_name, exp_dict) in enumerate(experiments, start=1):
        LOGGER.info("-" * 60)
        LOGGER.info("Experiment: %03d/%03d", exp_num, len(experiments))
        LOGGER.info("Filename: %s", exp_name)
        LOGGER.info("Task: %s", exp_dict.task_name)
        LOGGER.info("Model: %s", exp_dict.training_name)
        LOGGER.info("Method: %s", exp_dict.method_name)
        LOGGER.info("-" * 60)

        exp_dict.evaluation_name = get_experiment_name(
            exp_dict.method_name,
        )
        exp_dir = EXPERIMENTS_DIR.joinpath(
            'evaluation',
            exp_dict.task_name,
            exp_dict.training_name,
            exp_dict.evaluation_name,
        )
        exp_dir.mkdir(parents=True, exist_ok=True)
        exp_dir.joinpath('params.yaml').write_text(OmegaConf.to_yaml(exp_dict))

        exp_config = OmegaConf.to_object(exp_dict)

        main(
            config=exp_config,  # type: ignore[arg-type]
            save_dir=exp_dir,
        )
