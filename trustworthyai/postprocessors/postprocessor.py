from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import numpy as np
import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from trustworthyai.utils.metrics import calc_metrics

ModelType = TypeVar('ModelType', bound=pl.LightningModule)
DataModuleType = TypeVar('DataModuleType', bound=pl.LightningDataModule)


class Postprocessor(Generic[ModelType, DataModuleType], pl.LightningModule, ABC):
    def __init__(self, model: ModelType) -> None:
        super().__init__()
        self.model = model
        self.dataset_name: str

    def setup_postprocessor(
        self,
        outputs: list[dict[str, Tensor]],
    ) -> None:
        self.ind_predictions = [output['predictions'].detach().cpu() for output in outputs]
        self.ind_confidences = [output['confidences'].detach().cpu() for output in outputs]
        self.ind_labels = [output['labels'].detach().cpu() for output in outputs]

    def configure(
        self,
        datamodule: DataModuleType,
    ) -> None:
        pass

    @abstractmethod
    def create_dataloader(
        self,
        datamodule: DataModuleType,
        dataset_name: str,
    ) -> DataLoader:
        pass

    def training_step(  # type: ignore
        self,
        batch: dict[str, Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> dict[str, Tensor]:
        raise RuntimeError("Postprocessor has purpose only for prediction")

    def validation_step(  # type: ignore
        self,
        batch: dict[str, Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> dict[str, Tensor]:
        raise RuntimeError("Postprocessor has purpose only for prediction")

    def test_step(  # type: ignore
        self,
        batch: dict[str, Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> dict[str, Tensor]:
        return self.predict_step(
            batch=batch,
            batch_idx=batch_idx,
            dataloader_idx=dataloader_idx,
        )

    @abstractmethod
    def predict_step(  # type: ignore
        self,
        batch: dict[str, Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> dict[str, Tensor]:
        pass

    def test_epoch_end(self, outputs: list[dict[str, Tensor]]) -> None:  # type: ignore
        ood_predictions = [output['predictions'].detach().cpu() for output in outputs]
        ood_confidences = [output['confidences'].detach().cpu() for output in outputs]
        ood_labels = [output['labels'].detach().cpu() for output in outputs]

        predictions = torch.cat(self.ind_predictions + ood_predictions).numpy()
        confidences = torch.cat(self.ind_confidences + ood_confidences).numpy()
        labels = torch.cat(self.ind_labels + ood_labels).numpy()

        ground_truth = np.ones_like(labels)
        ground_truth[labels == -1] = 0

        metrics = calc_metrics(
            predictions=predictions,
            confidences=confidences,
            labels=labels,
            ground_truth=ground_truth,
        )

        self.log_dict({f'eval/{self.dataset_name}/{name}': val for name, val in metrics.items()})
