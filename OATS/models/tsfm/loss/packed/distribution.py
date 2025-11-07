# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import torch
from jaxtyping import Bool, Float, Int
from torch.distributions import Distribution

from ._base import PackedDistributionLoss


class PackedNLLLoss(PackedDistributionLoss):
    def _loss_func(
        self,
        pred: Distribution,
        target: Float[torch.Tensor, "*batch seq_len #dim"],
        prediction_mask: Bool[torch.Tensor, "*batch seq_len"],
        observed_mask: Bool[torch.Tensor, "*batch seq_len #dim"],
        sample_id: Int[torch.Tensor, "*batch seq_len"],
        variate_id: Int[torch.Tensor, "*batch seq_len"],
    ) -> Float[torch.Tensor, "*batch seq_len #dim"]:
        return -pred.log_prob(target)
