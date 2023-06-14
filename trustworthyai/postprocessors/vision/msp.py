import torch
from torch import Tensor

from trustworthyai.postprocessors.vision.postprocessor import VisionPostprocessor


class VisionMSP(VisionPostprocessor):
    @torch.no_grad()
    def forward(self, images: Tensor) -> tuple[Tensor, Tensor]:
        self.eval()
        logits = self.model(images)
        scores = torch.softmax(logits, dim=1)
        _, predictions = torch.max(scores, dim=1)
        confidences = torch.softmax(logits, dim=1).max(dim=-1).values
        return predictions, confidences
