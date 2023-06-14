import torch
from torch import Tensor
from torch.nn import Linear

from trustworthyai.postprocessors.vision.postprocessor import VisionPostprocessor
from trustworthyai.training.vision_classifier import VisionClassifier


class VisionGradNorm(VisionPostprocessor):
    def __init__(
        self,
        model: VisionClassifier,
    ) -> None:
        super().__init__(model=model)
        self.logsoftmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor]:
        fc = self.model.get_fc()
        self.model.eval()

        _, features = self.model(images, return_features=True)

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

            layer_grad_norm = torch.sum(torch.abs(fc.weight.grad.data))  # type: ignore[union-attr]
            confidences.append(layer_grad_norm)

        return torch.stack(confidences)
