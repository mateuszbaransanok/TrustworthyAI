from abc import ABC, abstractmethod

from torch import Tensor
from torch.utils.data import DataLoader

from trustworthyai.data.datasets.text_dataset import OODTextDataset
from trustworthyai.data.modules.transformer_datamodule import TransformerDataModule
from trustworthyai.postprocessors.postprocessor import Postprocessor
from trustworthyai.training.transformer_classifier import TransformerClassifier


class TransformerPostprocessor(Postprocessor[TransformerClassifier, TransformerDataModule], ABC):
    def __init__(
        self,
        model: TransformerClassifier,
    ) -> None:
        super().__init__(
            model=model,
        )

    @abstractmethod
    def forward(
        self,
        input_ids: Tensor,
        token_type_ids: Tensor,
        attention_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        pass

    def create_dataloader(
        self,
        datamodule: TransformerDataModule,
        dataset_name: str,
    ) -> DataLoader:
        self.dataset_name = dataset_name
        return DataLoader(
            dataset=OODTextDataset(
                name=dataset_name,
                split='test',
                tokenizer=datamodule.tokenize,
            ),
            shuffle=False,
            batch_size=datamodule.batch_size,
            num_workers=datamodule.num_workers,
            collate_fn=datamodule.collate,
            pin_memory=True,
        )

    def predict_step(
        self,
        batch: dict[str, Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> dict[str, Tensor]:
        labels = batch['labels']

        predictions, confidences = self(
            input_ids=batch['input_ids'],
            token_type_ids=batch['token_type_ids'],
            attention_mask=batch['attention_mask'],
        )

        return {
            'predictions': predictions,
            'confidences': confidences,
            'labels': labels,
        }
