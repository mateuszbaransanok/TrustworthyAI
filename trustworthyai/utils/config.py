import re
from datetime import datetime
from itertools import product
from typing import cast

from omegaconf import DictConfig, OmegaConf

from trustworthyai.settings import PROJECT_DIR


def get_experiment_name(*names: str) -> str:
    lst = [re.sub(r'[ _/]+', '-', name).lower() for name in names]
    lst.append(datetime.now().strftime('%y%m%d%H%M%S'))
    return '_'.join(lst)


def load_experiments(cls: type, *paths: str) -> list[DictConfig]:
    schema = OmegaConf.structured(cls)
    cli = OmegaConf.from_cli()
    config = cast(DictConfig, OmegaConf.merge(*(OmegaConf.load(PROJECT_DIR / p) for p in paths)))
    if 'experiments' in config:
        experiments = config.pop('experiments')
        return [
            cast(DictConfig, OmegaConf.merge(schema, config, cli))
            for config in (
                OmegaConf.merge(config, *experiment_config)
                for experiment in experiments
                for experiment_config in product(*experiment.values())
            )
        ]
    return [cast(DictConfig, OmegaConf.merge(schema, config, cli))]
