from abc import ABC
from typing import Literal

from torch.utils.data import Dataset

from trustworthyai.settings import DATASETS_DIR

TSplit = Literal['train', 'val', 'test', 'predict']


class BaseDataset(Dataset, ABC):
    TYPE: str

    def __init__(
        self,
        name: str,
        split: TSplit,
    ) -> None:
        self.name, _, self.version = name.partition('/')
        self.split = split

        self.classes: list[str]

        self.root_dir = DATASETS_DIR.joinpath(self.TYPE, self.name)
        self.annotation_dir = self.root_dir.joinpath('annotations', self.version)
