import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss

from trustworthyai.postprocessors.vision.postprocessor import VisionPostprocessor
from trustworthyai.training.vision_classifier import VisionClassifier


class VisionODIN(VisionPostprocessor):
    def __init__(
        self,
        model: VisionClassifier,
        temperature: float = 1000.0,
        epsilon: float = 0.0001,
    ) -> None:
        super().__init__(
            model=model,
        )
        self.temperature = temperature
        self.epsilon = epsilon
        self.criterion = CrossEntropyLoss()

    @torch.enable_grad()
    def forward(self, images: Tensor) -> tuple[Tensor, Tensor]:
        images.requires_grad = True

        outputs = self.model(images)

        max_labels = outputs.detach().argmax(axis=1)
        outputs = outputs / self.temperature

        loss = self.criterion(outputs, max_labels)
        loss.backward()

        # Normalizing the gradient to binary in {0, 1}
        gradient = torch.ge(images.grad, 0)  # type: ignore[arg-type]
        gradient = (gradient.float() - 0.5) * 2

        # Scaling values taken from original code
        gradient[:, 0] = gradient[:, 0] / (63.0 / 255.0)
        gradient[:, 1] = gradient[:, 1] / (62.1 / 255.0)
        gradient[:, 2] = gradient[:, 2] / (66.7 / 255.0)

        # Adding small perturbations to images
        perturbed_images = torch.add(images.detach(), gradient, alpha=-self.epsilon)
        outputs = self.model(perturbed_images)
        outputs = outputs / self.temperature

        outputs = outputs.detach()
        outputs = outputs - outputs.max(dim=1, keepdims=True).values
        outputs = outputs.exp() / outputs.exp().sum(dim=1, keepdims=True)

        confidences, predictions = outputs.max(dim=1)

        return predictions, confidences
