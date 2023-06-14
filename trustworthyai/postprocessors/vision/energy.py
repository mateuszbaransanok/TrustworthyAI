import torch
from torch import Tensor

from trustworthyai.postprocessors.vision.postprocessor import VisionPostprocessor
from trustworthyai.training.vision_classifier import VisionClassifier


class VisionEnergy(VisionPostprocessor):
    def __init__(
        self,
        model: VisionClassifier,
        temperature: float = 1.0,
    ) -> None:
        super().__init__(model=model)
        self.temperature = temperature

    @torch.no_grad()
    def forward(self, images: Tensor) -> tuple[Tensor, Tensor]:
        self.eval()
        logits = self.model(images)
        scores = torch.softmax(logits, dim=1)
        _, predictions = torch.max(scores, dim=1)
        confidences = self.temperature * torch.logsumexp(logits / self.temperature, dim=1)
        return predictions, confidences
