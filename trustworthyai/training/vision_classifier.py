from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss
from transformers import get_linear_schedule_with_warmup

from trustworthyai.models.vision.resnet18_32x32 import resnet18
from trustworthyai.training.classifier import BaseClassifier
from trustworthyai.utils.optimizers import OPTIMIZERS


class VisionClassifier(BaseClassifier):
    MODELS = {
        'resnet18': resnet18,
    }

    def __init__(
        self,
        model_name: str,
        classes: list[str],
        optimizer_name: str = 'adam',
        learning_rate: float = 2e-5,
        weight_decay: float = 5e-4,
        num_warmup_steps: int = 200,
    ) -> None:
        super().__init__(
            classes=classes,
        )
        self.save_hyperparameters(logger=False)

        self.optimizer_name = optimizer_name
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_warmup_steps = num_warmup_steps

        self.model = self.MODELS[model_name](
            pretrained=True,
            num_classes=self.num_classes,
        )

        self.criterion = CrossEntropyLoss()

    def forward(
        self,
        images: Tensor,
        return_features: bool = False,
    ) -> Tensor:
        return self.model(
            images=images,
            return_features=return_features,
        )

    def get_fc(self) -> nn.Linear:
        return self.model.get_fc()

    def step(
        self,
        batch: dict[str, Tensor],
        stage: str,
    ) -> dict[str, Tensor]:
        features = batch['image']
        labels = batch['label']
        logits = self(features)
        loss = self.criterion(logits, labels)
        return {
            'loss': loss,
            'labels': labels,
            'logits': logits,
        }

    def predict_step(
        self,
        batch: Tensor | dict[str, Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> list[str]:
        features = batch['image'] if isinstance(batch, dict) else batch
        logits = self(features)
        predictions = torch.argmax(logits, dim=-1)
        return [self.classes[pred] for pred in predictions]

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = OPTIMIZERS[self.optimizer_name](
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            params=self.parameters(),
        )
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=self.trainer.num_training_batches,  # type: ignore
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler,
        }
