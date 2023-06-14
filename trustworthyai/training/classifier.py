from abc import ABC, abstractmethod
from collections import Counter

import pytorch_lightning as pl
import torch
from torch import Tensor, nn
from torchmetrics import Accuracy, F1Score, MetricCollection, Precision, Recall


class BaseClassifier(pl.LightningModule, ABC):
    def __init__(
        self,
        classes: list[str],
    ) -> None:
        super().__init__()
        self.classes = classes
        self.num_classes = len(classes)

        metrics = MetricCollection(
            {
                'accuracy': Accuracy(),
                'f1_class': F1Score(num_classes=self.num_classes, average='none'),
                'f1': F1Score(num_classes=self.num_classes, average='macro'),
                'precision': Precision(num_classes=self.num_classes, average='macro'),
                'recall': Recall(num_classes=self.num_classes, average='macro'),
            }
        )
        self.metrics = nn.ModuleDict(
            {
                f'{stage}_metrics': metrics.clone(prefix=f'{stage}/')
                for stage in ('train', 'val', 'test')
            }
        )

    @abstractmethod
    def step(
        self,
        batch: dict[str, Tensor],
        stage: str,
    ) -> dict[str, Tensor]:
        pass

    def training_step(  # type: ignore
        self,
        batch: dict[str, Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> dict[str, Tensor]:
        return self.step(batch, 'train')

    def validation_step(  # type: ignore
        self,
        batch: dict[str, Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> dict[str, Tensor]:
        return self.step(batch, 'val')

    def test_step(  # type: ignore
        self,
        batch: dict[str, Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> dict[str, Tensor]:
        return self.step(batch, 'test')

    def training_epoch_end(  # type: ignore
        self,
        outputs: list[dict[str, Tensor]],
    ) -> None:
        self.log_epoch_metrics(outputs, stage='train')

    def validation_epoch_end(  # type: ignore
        self,
        outputs: list[dict[str, Tensor]],
    ) -> None:
        self.log_epoch_metrics(outputs, stage='val')

    def test_epoch_end(  # type: ignore
        self,
        outputs: list[dict[str, Tensor]],
    ) -> None:
        self.log_epoch_metrics(outputs, stage='test')

    def log_epoch_metrics(
        self,
        outputs: list[dict[str, Tensor]],
        stage: str,
    ) -> None:
        logits = torch.cat([out['logits'] for out in outputs])
        labels = torch.cat([out['labels'] for out in outputs])
        losses = torch.tensor([out['loss'] for out in outputs])

        metrics = self.metrics[f'{stage}_metrics'](logits, labels)
        f1_class = metrics.pop(f'{stage}/f1_class')
        classes = [f'{stage}/f1/{cls}' for cls in self.classes]
        metrics.update(zip(classes, f1_class, strict=True))

        self.log_dict(metrics, on_epoch=True, on_step=False)
        self.log(f'{stage}/loss', value=losses.mean(), on_epoch=True, on_step=False)
        self.log(f'{stage}/support', value=float(len(labels)), on_epoch=True, on_step=False)
        counter = Counter(self.classes[label.item()] for label in labels)
        support = {f'{stage}/support/{cls}': float(counter[cls]) for cls in self.classes}
        self.log_dict(support, on_epoch=True, on_step=False)
