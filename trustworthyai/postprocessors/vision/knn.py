import faiss
import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

from trustworthyai.data.modules.vision_datamodule import VisionDataModule
from trustworthyai.postprocessors.vision.postprocessor import VisionPostprocessor
from trustworthyai.training.vision_classifier import VisionClassifier


class VisionKNN(VisionPostprocessor):
    def __init__(
        self,
        model: VisionClassifier,
        k: int = 1,
        epsilon: float = 1e-10,
    ) -> None:
        super().__init__(
            model=model,
        )
        self.k = k
        self.epsilon = epsilon

    def _normalizer(self, x: np.ndarray) -> np.ndarray:
        return x / (np.linalg.norm(x, axis=-1, keepdims=True) + self.epsilon)

    def configure(
        self,
        datamodule: VisionDataModule,
    ) -> None:
        self.eval()

        features_list = []
        with torch.no_grad():
            for batch in tqdm(datamodule.train_dataloader(), desc="Configuring KNN"):
                images = batch['image'].to(self.device)
                _, features = self.model(images, return_features=True)
                features_list.append(self._normalizer(features.detach().cpu().numpy()))

        features_array = np.vstack(features_list)
        self.index = faiss.IndexFlatL2(features_array.shape[1])
        self.index.add(features_array)

    @torch.no_grad()
    def forward(self, images: Tensor) -> tuple[Tensor, Tensor]:
        self.eval()
        logits, features = self.model(images, return_features=True)
        features_normed = self._normalizer(features.detach().cpu().numpy())
        distances, _ = self.index.search(
            features_normed,
            k=self.k,
        )
        kth_dist = -distances[:, -1]
        _, predictions = torch.max(torch.softmax(logits, dim=1), dim=1)
        return predictions, torch.from_numpy(kth_dist)
