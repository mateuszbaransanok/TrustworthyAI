import numpy as np
import torch
from scipy.special import softmax
from sklearn.metrics import pairwise_distances_argmin_min
from torch import Tensor
from tqdm import tqdm

from trustworthyai.data.modules.vision_datamodule import VisionDataModule
from trustworthyai.postprocessors.vision.postprocessor import VisionPostprocessor
from trustworthyai.training.vision_classifier import VisionClassifier


class VisionKLMatching(VisionPostprocessor):
    def __init__(
        self,
        model: VisionClassifier,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__(
            model=model,
        )
        self.epsilon = epsilon

    def configure(
        self,
        datamodule: VisionDataModule,
    ) -> None:
        self.eval()

        logits_list = []
        with torch.no_grad():
            for batch in tqdm(datamodule.val_dataloader(), desc="Configuring KLMatching"):
                images = batch['image'].to(self.device)
                logits = self.model(images)
                logits_list.append(logits)

            logits_tensor = torch.vstack(logits_list)
            probabilities = softmax(logits_tensor.detach().cpu().numpy(), axis=-1)
            labels = np.argmax(probabilities, axis=-1)
            self.mean_probabilities = [
                np.nan_to_num(probabilities[labels == i].mean(axis=0))
                for i in tqdm(range(self.model.num_classes))
            ]

    def kl(self, p: np.ndarray, q: np.ndarray) -> np.ndarray:
        return np.sum(np.where(p != 0, p * np.log(p / (q + self.epsilon)), 0))

    @torch.no_grad()
    def forward(self, images: Tensor) -> tuple[Tensor, Tensor]:
        self.eval()
        logits = self.model(images)
        probabilities = softmax(logits.detach().cpu().numpy(), axis=-1)
        _, predictions = torch.max(logits, dim=1)
        scores = -pairwise_distances_argmin_min(
            X=probabilities,
            Y=np.array(self.mean_probabilities),
            metric=self.kl,
        )[1]
        return predictions, torch.from_numpy(scores)
