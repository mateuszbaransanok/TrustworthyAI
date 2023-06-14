from abc import ABC, abstractmethod

from torch import Tensor, nn


class VisionModel(nn.Module, ABC):
    @abstractmethod
    def forward(
        self,
        images: Tensor,
        return_features: bool = False,
    ) -> Tensor:
        pass

    @abstractmethod
    def get_fc(self) -> nn.Linear:
        pass
