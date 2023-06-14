import numpy as np
import torch
from numpy.linalg import norm, pinv
from scipy.special import logsumexp
from sklearn.covariance import EmpiricalCovariance
from torch import Tensor
from tqdm import tqdm

from trustworthyai.data.modules.transformer_datamodule import TransformerDataModule
from trustworthyai.postprocessors.transformer.postprocessor import (
    TransformerPostprocessor,
)
from trustworthyai.training.transformer_classifier import TransformerClassifier


class TransformerVIM(TransformerPostprocessor):
    def __init__(
        self,
        model: TransformerClassifier,
        dim: int = 42,
    ) -> None:
        super().__init__(
            model=model,
        )
        self.dim = dim

    def configure(
        self,
        datamodule: TransformerDataModule,
    ) -> None:
        fc = self.model.get_fc()

        features_list = []
        with torch.no_grad():
            for batch in tqdm(datamodule.train_dataloader(), desc="Configuring VIM"):
                _, features = self.model(
                    input_ids=batch['input_ids'].to(self.device),
                    token_type_ids=batch['token_type_ids'].to(self.device),
                    attention_mask=batch['attention_mask'].to(self.device),
                    return_features=True,
                )
                features_list.append(features)

        features_tensor = torch.vstack(features_list)
        features_array = features_tensor.detach().cpu().numpy()
        logits = fc(features_tensor).detach().cpu().numpy()

        self.u = -np.matmul(pinv(fc.weight.detach().cpu().numpy()), fc.bias.detach().cpu().numpy())
        ec = EmpiricalCovariance(assume_centered=True)
        ec.fit(features_array - self.u)
        eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
        self.NS = np.ascontiguousarray((eigen_vectors.T[np.argsort(eig_vals * -1)[self.dim :]]).T)
        vlogits = norm(np.matmul(features_array - self.u, self.NS), axis=-1)
        self.alpha = logits.max(axis=-1).mean() / vlogits.mean()

    @torch.no_grad()
    def forward(
        self,
        input_ids: Tensor,
        token_type_ids: Tensor,
        attention_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        self.eval()
        logits, features = self.model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            return_features=True,
        )
        features_array = features.detach().cpu().numpy()
        _, predictions = torch.max(logits, dim=1)
        energy = logsumexp(logits.detach().cpu().numpy(), axis=-1)
        vlogits = norm(np.matmul(features_array - self.u, self.NS), axis=-1) * self.alpha
        scores = -vlogits + energy  # type: ignore
        return predictions, torch.from_numpy(scores)
