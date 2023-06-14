import torch
from torch import Tensor

from trustworthyai.postprocessors.transformer.postprocessor import (
    TransformerPostprocessor,
)


class TransformerMSP(TransformerPostprocessor):
    @torch.no_grad()
    def forward(
        self,
        input_ids: Tensor,
        token_type_ids: Tensor,
        attention_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        self.eval()
        logits = self.model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )
        scores = torch.softmax(logits, dim=1)
        _, predictions = torch.max(scores, dim=1)
        confidences = torch.softmax(logits, dim=1).max(dim=-1).values
        return predictions, confidences
