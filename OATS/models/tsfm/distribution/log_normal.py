# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from typing import Callable

import torch
from jaxtyping import Float, PyTree
from torch.distributions import LogNormal
from torch.nn import functional as F

from ._base import DistributionOutput


class LogNormalOutput(DistributionOutput):
    distr_cls = LogNormal
    args_dim = dict(loc=1, scale=1)

    @property
    def domain_map(
        self,
    ) -> PyTree[
        Callable[[Float[torch.Tensor, "*batch 1"]], Float[torch.Tensor, "*batch"]], "T"
    ]:
        return dict(loc=self._loc, scale=self._scale)

    @staticmethod
    def _loc(loc: Float[torch.Tensor, "*batch 1"]) -> Float[torch.Tensor, "*batch"]:
        return loc.squeeze(-1)

    @staticmethod
    def _scale(scale: Float[torch.Tensor, "*batch 1"]) -> Float[torch.Tensor, "*batch"]:
        epsilon = torch.finfo(scale.dtype).eps
        return F.softplus(scale).clamp_min(epsilon).squeeze(-1)
