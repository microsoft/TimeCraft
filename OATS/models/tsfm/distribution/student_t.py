# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from typing import Callable

import torch
from jaxtyping import Float, PyTree
from torch.distributions import StudentT
from torch.nn import functional as F

from ._base import DistributionOutput


class StudentTOutput(DistributionOutput):
    distr_cls = StudentT
    args_dim = dict(df=1, loc=1, scale=1)

    @property
    def domain_map(
        self,
    ) -> PyTree[
        Callable[[Float[torch.Tensor, "*batch 1"]], Float[torch.Tensor, "*batch"]], "T"
    ]:
        return dict(df=self._df, loc=self._loc, scale=self._scale)

    @staticmethod
    def _df(df: Float[torch.Tensor, "*batch 1"]) -> Float[torch.Tensor, "*batch"]:
        return (2.0 + F.softplus(df)).squeeze(-1)

    @staticmethod
    def _loc(loc: Float[torch.Tensor, "*batch 1"]) -> Float[torch.Tensor, "*batch"]:
        return loc.squeeze(-1)

    @staticmethod
    def _scale(scale: Float[torch.Tensor, "*batch 1"]) -> Float[torch.Tensor, "*batch"]:
        epsilon = torch.finfo(scale.dtype).eps
        return F.softplus(scale).clamp_min(epsilon).squeeze(-1)
