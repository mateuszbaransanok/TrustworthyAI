from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss
from transformers import AutoModel, get_linear_schedule_with_warmup

from trustworthyai.training.classifier import BaseClassifier
from trustworthyai.utils.optimizers import OPTIMIZERS


class TransformerClassifier(BaseClassifier):
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

        self.model_name = model_name
        self.optimizer_name = optimizer_name
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_warmup_steps = num_warmup_steps

        self.language_model = AutoModel.from_pretrained(self.model_name)
        self.classification_model = nn.Linear(
            in_features=self.language_model.config.hidden_size,
            out_features=self.num_classes,
        )

        self.criterion = CrossEntropyLoss()

    def forward(
        self,
        input_ids: Tensor,
        token_type_ids: Tensor,
        attention_mask: Tensor,
        return_features: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor] | tuple[Tensor, list[Tensor]]:
        transformer_output = self.language_model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )
        embeddings = transformer_output.pooler_output
        output = self.classification_model(embeddings)

        if return_features:
            return output, embeddings

        return output

    def get_fc(self) -> nn.Linear:
        return self.classification_model

    def step(
        self,
        batch: dict[str, Tensor],
        stage: str,
    ) -> dict[str, Tensor]:
        labels = batch['labels']
        logits = self(
            input_ids=batch['input_ids'],
            token_type_ids=batch['token_type_ids'],
            attention_mask=batch['attention_mask'],
        )
        loss = self.criterion(logits, labels)
        return {
            'loss': loss,
            'labels': labels,
            'logits': logits,
        }

    def predict_step(
        self,
        batch: dict[str, Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> list[str]:
        logits = self(
            input_ids=batch['input_ids'],
            token_type_ids=batch['token_type_ids'],
            attention_mask=batch['attention_mask'],
        )
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
