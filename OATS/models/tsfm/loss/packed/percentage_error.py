# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import torch
from jaxtyping import Bool, Float, Int
from torch.nn import functional as F

from tsfm.common.torch_util import safe_div

from ._base import PackedPointLoss


class PackedMAPELoss(PackedPointLoss):
    def _loss_func(
        self,
        pred: Float[torch.Tensor, "*batch seq_len #dim"],
        target: Float[torch.Tensor, "*batch seq_len #dim"],
        prediction_mask: Bool[torch.Tensor, "*batch seq_len"],
        observed_mask: Bool[torch.Tensor, "*batch seq_len #dim"],
        sample_id: Int[torch.Tensor, "*batch seq_len"],
        variate_id: Int[torch.Tensor, "*batch seq_len"],
    ) -> Float[torch.Tensor, "*batch seq_len #dim"]:
        loss = F.l1_loss(pred, target, reduction="none")
        loss = safe_div(loss, target.abs())
        return 100 * loss


class PackedSMAPELoss(PackedPointLoss):
    def _loss_func(
        self,
        pred: Float[torch.Tensor, "*batch seq_len #dim"],
        target: Float[torch.Tensor, "*batch seq_len #dim"],
        prediction_mask: Bool[torch.Tensor, "*batch seq_len"],
        observed_mask: Bool[torch.Tensor, "*batch seq_len #dim"],
        sample_id: Int[torch.Tensor, "*batch seq_len"],
        variate_id: Int[torch.Tensor, "*batch seq_len"],
    ) -> Float[torch.Tensor, "*batch seq_len #dim"]:
        loss = F.l1_loss(pred, target, reduction="none")
        loss = safe_div(loss, target.abs() + pred.detach().abs())
        return 200 * loss
