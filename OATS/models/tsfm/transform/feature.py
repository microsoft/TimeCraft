# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import dataclass
from typing import Any

import numpy as np
from einops import repeat

from ._base import Transformation
from ._mixin import CheckArrNDimMixin, CollectFuncMixin


@dataclass
class AddVariateIndex(CollectFuncMixin, CheckArrNDimMixin, Transformation):
    """
    Add variate_id to data_entry
    """

    fields: tuple[str, ...]
    max_dim: int
    optional_fields: tuple[str, ...] = tuple()
    variate_id_field: str = "variate_id"
    expected_ndim: int = 2
    randomize: bool = False
    collection_type: type = list

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        self.counter = 0
        self.dimensions = (
            np.random.choice(self.max_dim, size=self.max_dim, replace=False)
            if self.randomize
            else list(range(self.max_dim))
        )
        data_entry[self.variate_id_field] = self.collect_func(
            self._generate_variate_id,
            data_entry,
            self.fields,
            optional_fields=self.optional_fields,
        )
        return data_entry

    def _generate_variate_id(
        self, data_entry: dict[str, Any], field: str
    ) -> np.ndarray:
        arr = data_entry[field]
        self.check_ndim(field, arr, self.expected_ndim)
        dim, time = arr.shape[:2]
        if self.counter + dim > self.max_dim:
            raise ValueError(
                f"Variate ({self.counter + dim}) exceeds maximum variate {self.max_dim}. "
            )
        field_dim_id = repeat(
            np.asarray(self.dimensions[self.counter : self.counter + dim], dtype=int),
            "var -> var time",
            time=time,
        )
        self.counter += dim
        return field_dim_id


@dataclass
class AddTimeIndex(CollectFuncMixin, CheckArrNDimMixin, Transformation):
    """
    Add time_id to data_entry
    """

    fields: tuple[str, ...]
    optional_fields: tuple[str, ...] = tuple()
    time_id_field: str = "time_id"
    expected_ndim: int = 2
    collection_type: type = list

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        """
        add sequence_id
        """
        data_entry[self.time_id_field] = self.collect_func(
            self._generate_time_id,
            data_entry,
            self.fields,
            optional_fields=self.optional_fields,
        )
        return data_entry

    def _generate_time_id(self, data_entry: dict[str, Any], field: str) -> np.ndarray:
        arr = data_entry[field]
        self.check_ndim(field, arr, self.expected_ndim)
        var, time = arr.shape[:2]
        field_seq_id = np.arange(time)
        field_seq_id = repeat(field_seq_id, "time -> var time", var=var)
        return field_seq_id


@dataclass
class AddObservedMask(CollectFuncMixin, Transformation):
    fields: tuple[str, ...]
    optional_fields: tuple[str, ...] = tuple()
    observed_mask_field: str = "observed_mask"
    collection_type: type = list

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        observed_mask = self.collect_func(
            self._generate_observed_mask,
            data_entry,
            self.fields,
            optional_fields=self.optional_fields,
        )
        data_entry[self.observed_mask_field] = observed_mask
        return data_entry

    @staticmethod
    def _generate_observed_mask(data_entry: dict[str, Any], field: str) -> np.ndarray:
        arr = data_entry[field]
        return ~np.isnan(arr)
