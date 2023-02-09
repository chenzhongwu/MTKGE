
from abc import ABC, abstractmethod
from typing import Tuple, Optional

import torch
from torch import nn


class Regularizer(nn.Module, ABC):
    @abstractmethod
    def forward(self, factor1: Tuple[torch.Tensor],factor2: Tuple[torch.Tensor],factor3: Tuple[torch.Tensor]):
        pass

class N3(Regularizer):
    def __init__(self, weight: float):
        super(N3, self).__init__()
        self.weight = weight

    def forward(self, factor1, factor2, factor3):
        norm = 0
        for f in factor1:
            norm += self.weight * torch.sum(torch.abs(f) ** 3)
        for f in factor2:
            norm += self.weight * torch.sum(torch.abs(f) ** 3)
        for f in factor3:
            norm += self.weight * torch.sum(torch.abs(f) ** 3)
        return norm / (factor1[0].shape[0]+factor2[0].shape[0]+factor3[0].shape[0])


# class Lambda3(Regularizer):
#     def __init__(self, weight: float):
#         super(Lambda3, self).__init__()
#         self.weight = weight
#
#     def forward(self, factor):
#         ddiff = factor[1:] - factor[:-1]
#         rank = int(ddiff.shape[1] / 2)
#         diff = torch.sqrt(ddiff[:, :rank]**2 + ddiff[:, rank:]**2)**3
#         return self.weight * torch.sum(diff) / (factor.shape[0] - 1)
