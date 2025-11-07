# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np

from tsfm.common.sampler import Sampler, get_sampler
from tsfm.common.typing import UnivarTimeSeries

from ._base import Transformation
from ._mixin import CheckArrNDimMixin, CollectFuncMixin, MapFuncMixin


@dataclass
class SampleDimension(
    CheckArrNDimMixin, CollectFuncMixin, MapFuncMixin, Transformation
):
    max_dim: int
    fields: tuple[str, ...]
    optional_fields: tuple[str, ...] = tuple()
    sampler: Sampler = get_sampler("uniform")

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        total_field_dim = sum(
            self.collect_func_list(
                self._get_dim,
                data_entry,
                self.fields,
                optional_fields=self.optional_fields,
            )
        )
        self.map_func(
            partial(self._process, total_field_dim=total_field_dim),  # noqa
            data_entry,
            self.fields,
            optional_fields=self.optional_fields,
        )
        return data_entry

    def _get_dim(self, data_entry: dict[str, Any], field: str) -> int:
        self.check_ndim(field, data_entry[field], 2)
        return len(data_entry[field])

    def _process(
        self, data_entry: dict[str, Any], field: str, total_field_dim: int
    ) -> list[UnivarTimeSeries]:
        arr: list[UnivarTimeSeries] = data_entry[field]
        rand_idx = np.random.permutation(len(arr))
        field_max_dim = (self.max_dim * len(arr)) // total_field_dim
        n = self.sampler(min(len(arr), field_max_dim))
        return [arr[idx] for idx in rand_idx[:n]]


@dataclass
class Subsample(Transformation):  # just take every n-th element
    fields: tuple[str, ...] = ("target", "past_feat_dynamic_real")

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        pass


class GaussianFilterSubsample(
    Subsample
):  # blur using gaussian filter before subsampling
    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        # gaussian filter
        return super()(data_entry)


class Downsample(Transformation):  # aggregate
    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        pass


class Upsample(Transformation):
    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        pass
