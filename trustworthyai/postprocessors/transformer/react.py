import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

from trustworthyai.data.modules.transformer_datamodule import TransformerDataModule
from trustworthyai.postprocessors.transformer.postprocessor import (
    TransformerPostprocessor,
)
from trustworthyai.training.transformer_classifier import TransformerClassifier


class TransformerReAct(TransformerPostprocessor):
    def __init__(
        self,
        model: TransformerClassifier,
        percentile: int = 90,
    ) -> None:
        super().__init__(
            model=model,
        )
        self.percentile = percentile

    def configure(
        self,
        datamodule: TransformerDataModule,
    ) -> None:
        self.eval()

        features_list = []
        with torch.no_grad():
            for batch in tqdm(datamodule.val_dataloader(), desc="Configuring ReAct"):
                _, features = self.model(
                    input_ids=batch['input_ids'].to(self.device),
                    token_type_ids=batch['token_type_ids'].to(self.device),
                    attention_mask=batch['attention_mask'].to(self.device),
                    return_features=True,
                )
                features_list.append(features.detach().cpu().numpy())

        self.threshold = np.percentile(np.vstack(features_list).flatten(), self.percentile)

    @torch.no_grad()
    def forward(
        self,
        input_ids: Tensor,
        token_type_ids: Tensor,
        attention_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        self.eval()
        _, features = self.model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            return_features=True,
        )

        features = features.clip(max=self.threshold)
        logits = self.model.get_fc()(features)

        scores = torch.softmax(logits, dim=1)
        _, predictions = torch.max(scores, dim=1)
        confidences = torch.logsumexp(logits.detach().cpu(), dim=1)
        return predictions, confidences
