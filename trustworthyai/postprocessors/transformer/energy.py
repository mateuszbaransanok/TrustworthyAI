import torch
from torch import Tensor

from trustworthyai.postprocessors.transformer.postprocessor import (
    TransformerPostprocessor,
)
from trustworthyai.training.transformer_classifier import TransformerClassifier


class TransformerEnergy(TransformerPostprocessor):
    def __init__(
        self,
        model: TransformerClassifier,
        temperature: float = 1.0,
    ) -> None:
        super().__init__(model=model)
        self.temperature = temperature

    @torch.no_grad()
    def forward(
        self,
        input_ids: Tensor,
        token_type_ids: Tensor,
        attention_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        self.eval()
        logits = self.model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )
        scores = torch.softmax(logits, dim=1)
        _, predictions = torch.max(scores, dim=1)
        confidences = self.temperature * torch.logsumexp(logits / self.temperature, dim=1)
        return predictions, confidences
