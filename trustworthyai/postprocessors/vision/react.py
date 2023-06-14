import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

from trustworthyai.data.modules.vision_datamodule import VisionDataModule
from trustworthyai.postprocessors.vision.postprocessor import VisionPostprocessor
from trustworthyai.training.vision_classifier import VisionClassifier


class VisionReAct(VisionPostprocessor):
    def __init__(
        self,
        model: VisionClassifier,
        percentile: int = 90,
    ) -> None:
        super().__init__(
            model=model,
        )
        self.percentile = percentile

    def configure(
        self,
        datamodule: VisionDataModule,
    ) -> None:
        self.eval()

        features_list = []
        with torch.no_grad():
            for batch in tqdm(datamodule.val_dataloader(), desc="Configuring ReAct"):
                images = batch['image'].to(self.device)
                _, features = self.model(images, return_features=True)
                features_list.append(features.detach().cpu().numpy())

        self.threshold = np.percentile(np.vstack(features_list).flatten(), self.percentile)

    @torch.no_grad()
    def forward(self, images: Tensor) -> tuple[Tensor, Tensor]:
        self.eval()
        _, features = self.model(images, return_features=True)

        features = features.clip(max=self.threshold)
        logits = self.model.get_fc()(features)

        scores = torch.softmax(logits, dim=1)
        _, predictions = torch.max(scores, dim=1)
        confidences = torch.logsumexp(logits.detach().cpu(), dim=1)
        return predictions, confidences
