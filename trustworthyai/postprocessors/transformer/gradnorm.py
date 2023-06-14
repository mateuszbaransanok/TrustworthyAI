import torch
from torch import Tensor
from torch.nn import Linear

from trustworthyai.postprocessors.transformer.postprocessor import (
    TransformerPostprocessor,
)
from trustworthyai.training.transformer_classifier import TransformerClassifier


class TransformerGradNorm(TransformerPostprocessor):
    def __init__(
        self,
        model: TransformerClassifier,
    ) -> None:
        super().__init__(model=model)
        self.logsoftmax = torch.nn.LogSoftmax(dim=-1)

    def forward(
        self,
        input_ids: Tensor,
        token_type_ids: Tensor,
        attention_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        self.eval()
        fc = self.model.get_fc()

        _, features = self.model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            return_features=True,
        )

        with torch.enable_grad():
            confidences = self.gradnorm(fc, features)

        with torch.no_grad():
            logits = fc(features)
            _, predictions = torch.max(logits, dim=1)

        return predictions, confidences

    def gradnorm(
        self,
        fc: Linear,
        features: Tensor,
    ) -> Tensor:
        confidences = []

        for feature in features:
            fc.zero_grad()
            logits = fc(feature.unsqueeze(0))

            ones = torch.ones(1, self.model.num_classes, device=self.device)  # type: ignore[call-overload]
            loss = torch.mean(torch.sum(-ones * self.logsoftmax(logits), dim=-1))
            loss.backward()

            layer_grad_norm = torch.sum(torch.abs(fc.weight.grad))  # type: ignore[arg-type]
            confidences.append(layer_grad_norm)

        return torch.stack(confidences)
