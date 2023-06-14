from abc import abstractmethod
from typing import Generic, TypeVar

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from trustworthyai.data.datasets.dataset import BaseDataset, TSplit

TDataset = TypeVar('TDataset', bound=BaseDataset)


class BaseDataModule(Generic[TDataset], pl.LightningDataModule):
    def __init__(
        self,
        dataset_name: str,
        batch_size: int = 32,
        num_workers: int = 6,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_name = dataset_name
        self.datasets: dict[str, TDataset] = {}

    def setup(
        self,
        stage: str | None = None,
    ) -> None:
        if stage in (None, 'fit', 'validate'):
            self.datasets['train'] = self.create_dataset(split='train')
            self.datasets['val'] = self.create_dataset(split='val')

        if stage in (None, 'test'):
            self.datasets['test'] = self.create_dataset(split='test')

        if stage == 'predict':
            self.datasets['predict'] = self.create_dataset(split='predict')

    @property
    def classes(self) -> list[str]:
        return next(iter(self.datasets.values())).classes

    @abstractmethod
    def create_dataset(
        self,
        split: TSplit,
    ) -> TDataset:
        pass

    def create_dataloader(
        self,
        split: TSplit,
    ) -> DataLoader:
        return DataLoader(
            dataset=self.datasets[split],
            shuffle=True if split == 'train' else False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def train_dataloader(self) -> DataLoader:
        return self.create_dataloader('train')

    def val_dataloader(self) -> DataLoader:
        return self.create_dataloader('val')

    def test_dataloader(self) -> DataLoader:
        return self.create_dataloader('test')

    def predict_dataloader(self) -> DataLoader:
        return self.create_dataloader('predict')
