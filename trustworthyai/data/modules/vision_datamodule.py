from trustworthyai.data.datasets.dataset import TSplit
from trustworthyai.data.datasets.vision_dataset import VisionDataset
from trustworthyai.data.modules.datamodule import BaseDataModule
from trustworthyai.data.transforms.vision_transforms import AUGMENTATION, TRANSFORMS


class VisionDataModule(BaseDataModule[VisionDataset]):
    def __init__(
        self,
        dataset_name: str,
        batch_size: int = 32,
        num_workers: int = 6,
    ) -> None:
        super().__init__(
            dataset_name=dataset_name,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        self.save_hyperparameters(logger=False)
        self.augmentation = AUGMENTATION.get(dataset_name)
        self.transform = TRANSFORMS.get(dataset_name)

    def create_dataset(
        self,
        split: TSplit,
    ) -> VisionDataset:
        return VisionDataset(
            name=self.dataset_name,
            split=split,
            transform=(self.augmentation or self.transform) if split == 'train' else self.transform,
        )
