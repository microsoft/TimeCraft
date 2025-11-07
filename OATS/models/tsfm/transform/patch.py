# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
from dataclasses import dataclass
from functools import partial
from typing import Any, Optional

import numpy as np
import pandas as pd
from einops import rearrange
from gluonts.time_feature import norm_freq_str
from jaxtyping import Num

from tsfm.common.typing import UnivarTimeSeries

from ._base import Transformation
from ._mixin import MapFuncMixin


class PatchSizeConstraints(abc.ABC):
    @abc.abstractmethod
    def _get_boundaries(self, n: int, offset_name: str) -> tuple[int, int]: ...

    def __call__(self, freq: str) -> range:
        offset = pd.tseries.frequencies.to_offset(freq)
        start, stop = self._get_boundaries(offset.n, norm_freq_str(offset.name))
        return range(start, stop + 1)


@dataclass
class FixedPatchSizeConstraints(PatchSizeConstraints):
    start: int
    stop: Optional[int] = None

    def __post_init__(self):
        if self.stop is None:
            self.stop = self.start
        assert self.start <= self.stop

    def _get_boundaries(self, n: int, offset_name: str) -> tuple[int, int]:
        return self.start, self.stop


class DefaultPatchSizeConstraints(PatchSizeConstraints):
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
    DEFAULT_RANGES = {
        "S": (64, 128),  # 512s = 8.53min, 4096s = 68.26min
        "T": (32, 128),  # 64min = 1.07h, 512min = 8.53h
        "H": (32, 64),  # 128h = 5.33days
        "D": (16, 32),
        "B": (16, 32),
        "W": (16, 32),
        "M": (8, 32),
        "Q": (1, 8),
        "Y": (1, 8),
        "A": (1, 8),
    }

    def _get_boundaries(self, n: int, offset_name: str) -> tuple[int, int]:
        start, stop = self.DEFAULT_RANGES[offset_name]
        return start, stop


@dataclass
class GetPatchSize(Transformation):
    min_time_patches: int
    target_field: str = "target"
    patch_size: tuple[int, ...] | list[int] | range = (8, 16, 32, 64, 128)
    patch_size_constraints: PatchSizeConstraints = DefaultPatchSizeConstraints()
    offset: bool = True

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        data_entry["patch_size"] = np.int64(32)
        return data_entry


@dataclass
class Patchify(MapFuncMixin, Transformation):
    max_patch_size: int
    fields: tuple[str, ...] = ("target",)
    optional_fields: tuple[str, ...] = ("past_feat_dynamic_real",)
    pad_value: int | float = 0

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        patch_size = data_entry["patch_size"]
        self.map_func(
            partial(self._patchify, patch_size=patch_size),  # noqa
            data_entry,
            self.fields,
            optional_fields=self.optional_fields,
        )
        return data_entry

    def _patchify(self, data_entry: dict[str, Any], field: str, patch_size: int):
        arr = data_entry[field]
        if isinstance(arr, list):
            return [self._patchify_arr(a, patch_size) for a in arr]
        if isinstance(arr, dict):
            for k, v in arr.items():
                if k in self.fields or k in self.optional_fields:
                    arr[k] = self._patchify_arr(v, patch_size)
            return arr
        return self._patchify_arr(arr, patch_size)

    def _patchify_arr(
        self, arr: Num[np.ndarray, "var time*patch"], patch_size: int
    ) -> Num[np.ndarray, "var time max_patch"]:
        assert arr.shape[-1] % patch_size == 0
        arr = rearrange(arr, "... (time patch) -> ... time patch", patch=patch_size)
        pad_width = [(0, 0) for _ in range(arr.ndim)]
        pad_width[-1] = (0, self.max_patch_size - patch_size)
        arr = np.pad(arr, pad_width, mode="constant", constant_values=self.pad_value)
        return arr
