# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import dataclass
from typing import Any

import numpy as np

from ._base import Transformation
from ._mixin import MapFuncMixin


@dataclass
class Pad(MapFuncMixin, Transformation):
    min_length: int
    fields: tuple[str, ...]
    optional_fields: tuple[str, ...] = tuple()

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        self.map_func(
            self.map,
            data_entry,
            self.fields,
            optional_fields=self.optional_fields,
        )
        return data_entry

    def map(self, data_entry: dict[str, Any], field: str) -> Any:
        arr = data_entry[field]
        length = arr.shape[-1]
        if length < self.min_length:
            pad_amount = self.min_length - length
            front_pad = np.random.randint(0, pad_amount + 1)
            back_pad = pad_amount - front_pad
            pad_width = [(0, 0) for _ in range(arr.ndim)]
            pad_width[-1] = (front_pad, back_pad)
            arr = np.pad(arr, pad_width, mode="constant", constant_values=np.nan)
        return arr


@dataclass
class PadFreq(MapFuncMixin, Transformation):
    freq_min_length_map: dict[str, int]
    fields: tuple[str, ...]
    optional_fields: tuple[str, ...] = tuple()
    freq_field: str = "freq"

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        self.map_func(
            self.map,
            data_entry,
            self.fields,
            optional_fields=self.optional_fields,
        )
        return data_entry

    def map(self, data_entry: dict[str, Any], field: str) -> Any:
        arr = data_entry[field]
        length = arr.shape[-1]
        min_length = self.freq_min_length_map[data_entry[self.freq_field]]
        if length < min_length:
            pad_amount = min_length - length
            front_pad = np.random.randint(0, pad_amount + 1)
            back_pad = pad_amount - front_pad
            pad_width = [(0, 0) for _ in range(arr.ndim)]
            pad_width[-1] = (front_pad, back_pad)
            arr = np.pad(arr, pad_width, mode="constant", constant_values=np.nan)
        return arr


@dataclass
class EvalPad(MapFuncMixin, Transformation):
    prediction_pad: int
    context_pad: int
    fields: tuple[str, ...]
    optional_fields: tuple[str, ...] = tuple()

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        self.map_func(
            self.map,
            data_entry,
            self.fields,
            optional_fields=self.optional_fields,
        )
        return data_entry

    def map(self, data_entry: dict[str, Any], field: str) -> Any:
        arr = data_entry[field]
        pad_width = [(0, 0) for _ in range(arr.ndim)]
        pad_width[-1] = (self.context_pad, self.prediction_pad)
        arr = np.pad(arr, pad_width, mode="constant", constant_values=np.nan)
        return arr


@dataclass
class EvalPad_AdaLength(MapFuncMixin, Transformation):
    prediction_length: int
    context_length: int
    patch_size: int
    fields: tuple[str, ...]
    optional_fields: tuple[str, ...] = tuple()

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        self.map_func(
            self.map,
            data_entry,
            self.fields,
            optional_fields=self.optional_fields,
        )
        return data_entry

    def map(self, data_entry: dict[str, Any], field: str) -> Any:
        if self.context_length == -7777:
            context_length = data_entry["target"].shape[-1] - self.prediction_length
        else:
            context_length = self.context_length
        prediction_length = self.prediction_length

        context_pad = (-context_length) % self.patch_size
        prediction_pad = (-prediction_length) % self.patch_size
        
        arr = data_entry[field]
        pad_width = [(0, 0) for _ in range(arr.ndim)]
        pad_width[-1] = (context_pad, prediction_pad)
        arr = np.pad(arr, pad_width, mode="constant", constant_values=np.nan)
        return arr
