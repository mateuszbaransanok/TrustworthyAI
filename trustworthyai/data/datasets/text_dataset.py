from pathlib import Path
from typing import Any, Callable

from torch import Tensor

from trustworthyai.data.datasets.dataset import BaseDataset, TSplit


class TextDataset(BaseDataset):
    TYPE: str = 'text'

    def __init__(
        self,
        name: str,
        split: TSplit,
        tokenizer: Callable[[str], dict[str, Tensor]] | None = None,
    ) -> None:
        super().__init__(
            name=name,
            split=split,
        )
        self.tokenizer = tokenizer

        self._load_data()

    def _load_data(self) -> None:
        self.classes = self.annotation_dir.joinpath('classes.txt').read_text().splitlines()
        self.class_to_index = {cls: self.classes.index(cls) for cls in self.classes}
        self.index_to_class = {inx: cls for cls, inx in self.class_to_index.items()}

        self.items = []
        annotation_path = self.annotation_dir.joinpath(self.split).with_suffix('.txt')
        for item in annotation_path.read_text().splitlines():
            path, _, label = item.partition(' ')
            self.items.append((self.root_dir.joinpath(path), self.class_to_index[label]))

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Any]:
        path, label = self.items[index]

        text = self.load_text(path)

        if self.tokenizer is not None:
            data = self.tokenizer(text)
            return {
                **data,
                'label': label,
            }

        return {
            'text': text,
            'label': label,
        }

    def __len__(self) -> int:
        return len(self.items)

    @staticmethod
    def load_text(path: Path) -> str:
        return path.read_text()


class OODTextDataset(TextDataset):
    def _load_data(self) -> None:
        self.classes = ['ood']
        self.class_to_index = {'ood': -1}
        self.index_to_class = {-1: 'ood'}

        self.items = []
        annotation_path = self.annotation_dir.joinpath(self.split).with_suffix('.txt')
        for item in annotation_path.read_text().splitlines():
            path, _, label = item.partition(' ')
            self.items.append((self.root_dir.joinpath(path), -1))
