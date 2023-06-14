import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

from trustworthyai.data.modules.vision_datamodule import VisionDataModule
from trustworthyai.postprocessors.vision.postprocessor import VisionPostprocessor
from trustworthyai.training.vision_classifier import VisionClassifier


class VisionDICE(VisionPostprocessor):
    def __init__(
        self,
        model: VisionClassifier,
        p: int = 90,
    ) -> None:
        super().__init__(
            model=model,
        )
        self.p = p

    def configure(
        self,
        datamodule: VisionDataModule,
    ) -> None:
        self.eval()
        fc = self.model.get_fc()

        features_list = []
        with torch.no_grad():
            for batch in tqdm(datamodule.train_dataloader(), desc="Configuring DICE"):
                images = batch['image'].to(self.device)
                _, features = self.model(images, return_features=True)
                features_list.append(features.cpu().numpy())

        mean_features = np.vstack(features_list).mean(axis=0, keepdims=True)
        contribution = mean_features * fc.weight.detach().cpu().numpy()
        threshold = np.percentile(contribution, self.p)
        mask = torch.tensor((contribution > threshold), device=self.device)  # type: ignore[arg-type]
        self.masked_weight = fc.weight * mask
        self.bias = fc.bias

    @torch.no_grad()
    def forward(self, images: Tensor) -> tuple[Tensor, Tensor]:
        self.eval()
        _, features = self.model(images, return_features=True)
        vote = features[:, None, :] * self.masked_weight.to(self.device)  # type: ignore[arg-type]
        output = vote.sum(axis=2) + self.bias.to(self.device)  # type: ignore[arg-type]
        _, predictions = torch.max(torch.softmax(output, dim=1), dim=1)
        confidences = torch.logsumexp(output, dim=1)
        return predictions, confidences
