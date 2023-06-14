from pathlib import Path
from typing import Any, Callable

from PIL import Image
from torch import Tensor

from trustworthyai.data.datasets.dataset import BaseDataset, TSplit


class VisionDataset(BaseDataset):
    TYPE = 'vision'

    def __init__(
        self,
        name: str,
        split: TSplit,
        transform: Callable[[Image.Image], Tensor] | None = None,
    ) -> None:
        super().__init__(
            name=name,
            split=split,
        )
        self.transform = transform

        self._load_data()

    def _load_data(self) -> None:
        self.classes = self.annotation_dir.joinpath('classes.txt').read_text().splitlines()
        self.class_to_index = {cls: self.classes.index(cls) for cls in self.classes}
        self.index_to_class = {inx: cls for cls, inx in self.class_to_index.items()}

        self.items = []
        annotation_path = self.annotation_dir.joinpath(self.split).with_suffix('.txt')
        for item in annotation_path.read_text().splitlines():
            rel_path, _, label = item.partition(' ')
            abs_path = self.root_dir.joinpath(rel_path)
            index = self.class_to_index[label]
            self.items.append((abs_path, index))

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Any]:
        path, label = self.items[index]

        image = self.load_image(path)

        if self.transform is not None:
            image = self.transform(image)

        return {
            'image': image,
            'label': label,
        }

    def __len__(self) -> int:
        return len(self.items)

    @staticmethod
    def load_image(path: Path) -> Image.Image:
        with path.open('rb') as file:
            image = Image.open(file)
            return image.convert('RGB')


class OODVisionDataset(VisionDataset):
    def _load_data(self) -> None:
        self.classes = ['ood']
        self.class_to_index = {'ood': -1}
        self.index_to_class = {-1: 'ood'}

        self.items = []
        annotation_path = self.annotation_dir.joinpath(self.split).with_suffix('.txt')
        for item in annotation_path.read_text().splitlines():
            rel_path, _, label = item.partition(' ')
            abs_path = self.root_dir.joinpath(rel_path)
            self.items.append((abs_path, -1))
