from abc import ABC, abstractmethod

from torch import Tensor
from torch.utils.data import DataLoader

from trustworthyai.data.datasets.vision_dataset import OODVisionDataset
from trustworthyai.data.modules.vision_datamodule import VisionDataModule
from trustworthyai.postprocessors.postprocessor import Postprocessor
from trustworthyai.training.vision_classifier import VisionClassifier


class VisionPostprocessor(Postprocessor[VisionClassifier, VisionDataModule], ABC):
    def __init__(
        self,
        model: VisionClassifier,
    ) -> None:
        super().__init__(
            model=model,
        )

    def create_dataloader(
        self,
        datamodule: VisionDataModule,
        dataset_name: str,
    ) -> DataLoader:
        self.dataset_name = dataset_name
        return DataLoader(
            dataset=OODVisionDataset(
                name=dataset_name,
                split='test',
                transform=datamodule.transform,
            ),
            shuffle=False,
            batch_size=datamodule.batch_size,
            num_workers=datamodule.num_workers,
            pin_memory=True,
        )

    @abstractmethod
    def forward(self, images: Tensor) -> tuple[Tensor, Tensor]:
        pass

    def predict_step(
        self,
        batch: dict[str, Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> dict[str, Tensor]:
        images = batch['image']
        labels = batch['label']

        predictions, confidences = self(images)

        return {
            'predictions': predictions,
            'confidences': confidences,
            'labels': labels,
        }
