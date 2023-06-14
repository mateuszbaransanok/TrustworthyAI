import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from time import sleep
from typing import Any

from lightning_lite import seed_everything
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger

from trustworthyai.data.modules.datamodule import BaseDataModule
from trustworthyai.data.modules.transformer_datamodule import TransformerDataModule
from trustworthyai.data.modules.vision_datamodule import VisionDataModule
from trustworthyai.settings import EXPERIMENTS_DIR
from trustworthyai.training.classifier import BaseClassifier
from trustworthyai.training.transformer_classifier import TransformerClassifier
from trustworthyai.training.vision_classifier import VisionClassifier
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


@dataclass(kw_only=True)
class TrainerConfig:
    max_epochs: int
    gpu: list[int] | None


@dataclass(kw_only=True)
class TrainConfig:
    seed: int
    task_name: str
    training_name: str
    data_name: str
    data_config: dict[str, Any]
    model_name: str
    model_config: dict[str, Any]
    trainer: TrainerConfig


def main(config: TrainConfig, save_dir: Path) -> None:
    # SETUP ----------------------------------------------------------------------------------------
    seed_everything(config.seed)

    # DATAMODULE -----------------------------------------------------------------------------------
    datamodule = DATA_MODULES[config.task_name](**config.data_config)
    datamodule.prepare_data()
    datamodule.setup('fit')

    # MODEL ----------------------------------------------------------------------------------------
    model = MODELS[config.task_name](
        classes=datamodule.classes,
        **config.model_config,  # type: ignore
    )

    # TRAINER --------------------------------------------------------------------------------------
    checkpoint_dir = save_dir / 'checkpoints'
    resume_path = checkpoint_dir / 'last.ckpt'
    resume_ckpt = str(resume_path) if resume_path.exists() else None

    logger = WandbLogger(
        project='TrustworthyAI',
        name=save_dir.name,
        version=save_dir.name,
        group=save_dir.name,
        job_type='training',
        save_dir=str(save_dir),
        resume='allow',
    )
    logger.log_hyperparams(asdict(config))

    trainer = Trainer(
        max_epochs=config.trainer.max_epochs,
        accelerator='gpu' if bool(config.trainer.gpu) else 'cpu',
        precision=16,
        devices=config.trainer.gpu,
        logger=logger,
        callbacks=[
            ModelCheckpoint(
                monitor='val/f1',
                mode='max',
                save_last=True,
                save_top_k=1,
                dirpath=checkpoint_dir,
            ),
            EarlyStopping(
                monitor='val/f1',
                mode='max',
                patience=10,
            ),
            LearningRateMonitor(
                logging_interval='epoch',
            ),
        ],
    )

    try:
        # FITTING ----------------------------------------------------------------------------------
        trainer.fit(
            model=model,
            datamodule=datamodule,
            ckpt_path=resume_ckpt,
        )

        # TESTING ----------------------------------------------------------------------------------
        metrics = trainer.test(
            datamodule=datamodule,
            ckpt_path='best',
        )
    except Exception as exc:
        LOGGER.error("Experiment error", exc_info=exc)
        sleep(5)
    else:
        # SAVING METRICS ---------------------------------------------------------------------------
        with save_dir.joinpath('metrics.json').open('w') as file:
            json.dump(
                obj=metrics,
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
            'params/training/vision-cifar10.yaml',
            'params/training/transformer-news-category.yaml',
            'params/training/transformer-news-groups.yaml',
            'params/training/transformer-reviews.yaml',
        ]
        for experiment in load_experiments(TrainConfig, filename)
    ]

    for exp_num, (exp_name, exp_dict) in enumerate(experiments, start=1):
        LOGGER.info("-" * 60)
        LOGGER.info("Experiment: %03d/%03d", exp_num, len(experiments))
        LOGGER.info("Filename: %s", exp_name)
        LOGGER.info("Task: %s", exp_dict.task_name)
        LOGGER.info("Data: %s", exp_dict.data_name)
        LOGGER.info("Model: %s", exp_dict.model_name)
        LOGGER.info("-" * 60)

        exp_dict.training_name = get_experiment_name(
            exp_dict.data_name,
            exp_dict.model_name,
        )
        exp_dir = EXPERIMENTS_DIR / 'training' / exp_dict.task_name / exp_dict.training_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        exp_dir.joinpath('params.yaml').write_text(OmegaConf.to_yaml(exp_dict))

        exp_config = OmegaConf.to_object(exp_dict)

        main(
            config=exp_config,  # type: ignore[arg-type]
            save_dir=exp_dir,
        )
