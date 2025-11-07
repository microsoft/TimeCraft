# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import torch
import numpy as np
from einops import rearrange, reduce
from functools import partial
from typing import (
    Collection,
    Optional,
    Callable,
    Mapping,
    Dict,
    List,
    Iterator,
)
from gluonts.ev.stats import (
    error,
    absolute_error,
    absolute_label,
    absolute_percentage_error,
    absolute_scaled_error,
    coverage,
    quantile_loss,
    scaled_interval_score,
    scaled_quantile_loss,
    squared_error,
    symmetric_absolute_percentage_error,
    num_masked_target_values,
)
from gluonts.evaluation.metrics import calculate_seasonal_error


def mse(data: Dict[str, torch.Tensor], forecast_type: str = "0.5") -> torch.Tensor:
    # aggregate: mean
    return squared_error(data, forecast_type)


def mae(data: Dict[str, torch.Tensor], forecast_type: str = "0.5") -> torch.Tensor:
    # aggregate: mean
    return absolute_error(data, forecast_type)


def mase(data: Dict[str, torch.Tensor], forecast_type: str = "0.5") -> torch.Tensor:
    # aggregate: mean
    return absolute_scaled_error(data, forecast_type)


def mape(data: Dict[str, torch.Tensor], forecast_type: str = "0.5") -> torch.Tensor:
    # aggregate: mean
    return safe_div(absolute_error(data, forecast_type), absolute_label(data))


def smape(data: Dict[str, torch.Tensor], forecast_type: str = "0.5") -> torch.Tensor:
    # aggregate: mean
    # return symmetric_absolute_percentage_error(data, forecast_type)
    return safe_div(
        2 * absolute_error(data, forecast_type),
        (absolute_label(data) + np.abs(data[forecast_type])),
    )


def msis(data: Dict[str, torch.Tensor]) -> torch.Tensor:
    # aggregate: mean
    # data['seasonal_error']
    return scaled_interval_score(data, alpha=0.05)


def rmse(data: Dict[str, torch.Tensor], forecast_type: str = "mean") -> torch.Tensor:
    # aggregate: mean
    return np.sqrt(squared_error(data, forecast_type))


def nrmse(data: Dict[str, torch.Tensor], forecast_type: str = "mean") -> torch.Tensor:
    # aggregate: mean
    return safe_div(np.sqrt(squared_error(data, forecast_type)), absolute_label(data))


def nd(data: Dict[str, torch.Tensor], forecast_type: str = "0.5") -> torch.Tensor:
    # aggregate: sum
    return safe_div(absolute_error(data, forecast_type), absolute_label(data))


def mean_weighted_seq_quantile_loss(data: Dict[str, torch.Tensor]) -> torch.Tensor:
    # MeanWeightedSumQuantileLoss: aggregate: sum
    stacked_quantile_losses = []
    for q in np.arange(0.1, 1, 0.1):
        stacked_quantile_losses.append(
            safe_div(quantile_loss(data, q), absolute_label(data))
        )
    stacked_quantile_losses = np.stack(stacked_quantile_losses, axis=0)
    mean_quantile_loss = stacked_quantile_losses.mean(axis=0)
    return mean_quantile_loss


def safe_div(
    numer: torch.Tensor,
    denom: torch.Tensor,
) -> torch.Tensor:
    return numer / np.where(
        denom == 0,
        1.0,
        denom,
    )


class ValMetric:
    stat: Callable
    aggregate: str

    def __init__(self, stat, aggregate) -> None:
        self.stat = stat
        self.aggregate = aggregate

    def __call__(
        self, pred, target, observed_mask, prediction_mask, sample_id, variate_id
    ) -> torch.Tensor:
        """
        pred: torch.Tensor, shape (batch_size, patch_len, num_features)
        target: torch.Tensor, shape (batch_size, patch_len, num_features)
        observed_mask: torch.Tensor, shape (batch_size, patch_len, num_features)
        prediction_mask: torch.Tensor, shape (batch_size, patch_len)
        sample_id: torch.Tensor, shape (batch_size, patch_len)
        variate_id: torch.Tensor, shape (batch_size, patch_len)
        freq: str, shape batch_size
        """

        data = {}
        for q in np.arange(0.1, 1, 0.1):
            data[str(q)] = torch.quantile(pred, q=q, dim=1).cpu().numpy()
        data["mean"] = pred.mean(dim=1).cpu().numpy()
        data["label"] = target.cpu().numpy()
        data["seasonal_error"] = calculate_seasonal_error(data["label"], seasonality=1)

        id_mask = torch.logical_and(
            torch.eq(sample_id.unsqueeze(-1), sample_id.unsqueeze(-2)),
            torch.eq(variate_id.unsqueeze(-1), variate_id.unsqueeze(-2)),
        )
        mask = prediction_mask.unsqueeze(-1) * observed_mask
        tobs = reduce(
            id_mask
            * reduce(
                mask,
                "... seq dim -> ... 1 seq",
                "sum",
            ),
            "... seq1 seq2 -> ... seq1 1",
            "sum",
        )
        nobs = reduce(
            id_mask * rearrange(prediction_mask, "... seq -> ... 1 seq"),
            "... seq1 seq2 -> ... seq1 1",
            "sum",
        ) * prediction_mask.unsqueeze(-1)
        nobs = torch.where(nobs == 0, nobs, 1 / nobs).sum()

        tobs = tobs.cpu().numpy()
        nobs = nobs.cpu().numpy()
        mask = mask.cpu().numpy()

        metric = self.stat(data)

        if self.aggregate == "mean":
            metric = safe_div(metric, tobs * nobs)
        elif self.aggregate == "sum":
            pass
        metric = (metric * mask).sum()

        return metric


class MSE_mean(ValMetric):
    def __init__(self) -> None:
        stat = partial(mse, forecast_type="mean")
        aggregate = "mean"
        super().__init__(stat, aggregate)


class MAE_mean(ValMetric):
    def __init__(self) -> None:
        stat = partial(mae, forecast_type="mean")
        aggregate = "mean"
        super().__init__(stat, aggregate)


class MSE_median(ValMetric):
    def __init__(self) -> None:
        stat = partial(mse, forecast_type="0.5")
        aggregate = "mean"
        super().__init__(stat, aggregate)


class MAE_median(ValMetric):
    def __init__(self) -> None:
        stat = partial(mae, forecast_type="0.5")
        aggregate = "mean"
        super().__init__(stat, aggregate)


class MASE(ValMetric):
    def __init__(self) -> None:
        stat = mase
        aggregate = "mean"
        super().__init__(stat, aggregate)


class MAPE(ValMetric):
    def __init__(self) -> None:
        stat = mape
        aggregate = "mean"
        super().__init__(stat, aggregate)


class SMAPE(ValMetric):
    def __init__(self) -> None:
        stat = smape
        aggregate = "mean"
        super().__init__(stat, aggregate)


class RMSE(ValMetric):
    def __init__(self) -> None:
        stat = partial(rmse, forecast_type="mean")
        aggregate = "mean"
        super().__init__(stat, aggregate)


class NRMSE(ValMetric):
    def __init__(self) -> None:
        stat = partial(nrmse, forecast_type="mean")
        aggregate = "mean"
        super().__init__(stat, aggregate)


class ND(ValMetric):
    def __init__(self) -> None:
        stat = partial(nd, forecast_type="0.5")
        aggregate = "sum"
        super().__init__(stat, aggregate)


class CRPS(ValMetric):
    def __init__(self) -> None:
        stat = mean_weighted_seq_quantile_loss
        aggregate = "mean"
        super().__init__(stat, aggregate)
