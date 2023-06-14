from torch import Tensor
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding

from trustworthyai.data.datasets.dataset import TSplit
from trustworthyai.data.datasets.text_dataset import TextDataset
from trustworthyai.data.modules.datamodule import BaseDataModule


class TransformerDataModule(BaseDataModule[TextDataset]):
    def __init__(
        self,
        tokenizer_name: str,
        dataset_name: str,
        batch_size: int = 32,
        num_workers: int = 6,
        max_length: int = 512,
    ) -> None:
        super().__init__(
            dataset_name=dataset_name,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        self.save_hyperparameters(logger=False)
        self.tokenizer_name = tokenizer_name
        self.max_length = max_length
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self._tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
        self._collate = DataCollatorWithPadding(self._tokenizer)

    def collate(
        self,
        features: list[dict[str, Tensor]],
    ) -> dict[str, Tensor]:
        return self._collate(features)

    def tokenize(
        self,
        text: str,
    ) -> dict[str, Tensor]:
        return self._tokenizer(
            text=text,
            truncation=True,
            add_special_tokens=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            padding=False,
            max_length=self.max_length,
        )

    def create_dataset(
        self,
        split: TSplit,
    ) -> TextDataset:
        return TextDataset(
            name=self.dataset_name,
            split=split,
            tokenizer=self.tokenize,
        )

    def create_dataloader(
        self,
        split: str,
    ) -> DataLoader:
        return DataLoader(
            dataset=self.datasets[split],
            shuffle=True if split == 'train' else False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate,
            pin_memory=True,
        )
