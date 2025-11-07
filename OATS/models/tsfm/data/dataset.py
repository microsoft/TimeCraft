# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from enum import Enum
from typing import Any

import numpy as np
from torch.utils.data import Dataset

from tsfm.common.sampler import Sampler, get_sampler
from tsfm.common.typing import (
    BatchedData,
    BatchedDateTime,
    BatchedString,
    Data,
    FlattenedData,
    MultivarTimeSeries,
    UnivarTimeSeries,
)
from tsfm.data.indexer import Indexer
from tsfm.transform import Transformation


class SampleTimeSeriesType(Enum):
    NONE = "none"
    UNIFORM = "uniform"
    PROPORTIONAL = "proportional"


class TimeSeriesDataset(Dataset):
    def __init__(
        self,
        indexer: Indexer[dict[str, Any]],
        transform: Transformation,
        sample_time_series: SampleTimeSeriesType = SampleTimeSeriesType.NONE,
        dataset_weight: float = 1.0,
    ):
        self.indexer = indexer
        self.transform = transform
        self.sample_time_series = sample_time_series
        self.dataset_weight = dataset_weight

        if sample_time_series == SampleTimeSeriesType.NONE:
            self.probabilities = None
        elif sample_time_series == SampleTimeSeriesType.UNIFORM:
            self.probabilities = indexer.get_uniform_probabilities()
        elif sample_time_series == SampleTimeSeriesType.PROPORTIONAL:
            self.probabilities = indexer.get_proportional_probabilities()
        else:
            raise ValueError(f"Unknown sample type {sample_time_series}")

    def __getitem__(self, idx: int) -> dict[str, FlattenedData]:
        if idx < 0 or idx >= len(self):
            raise IndexError(
                f"Index {idx} out of range for dataset of length {len(self)}"
            )

        if self.sample_time_series != SampleTimeSeriesType.NONE:
            idx = np.random.choice(len(self.probabilities), p=self.probabilities)

        return self.transform(self._flatten_data(self._get_data(idx)))

    @property
    def num_ts(self) -> int:
        return len(self.indexer)

    def __len__(self) -> int:
        return int(np.ceil(self.num_ts * self.dataset_weight))

    def _get_data(self, idx: int) -> dict[str, Data | BatchedData]:
        return self.indexer[idx % self.num_ts]

    @staticmethod
    def _flatten_data(data: dict[str, Data]) -> dict[str, FlattenedData]:
        return {
            k: (
                [v]
                if isinstance(v, UnivarTimeSeries)
                else list(v) if isinstance(v, MultivarTimeSeries) else v
            )
            for k, v in data.items()
        }


class TimeSeriesDatasetWithIndex(TimeSeriesDataset):
    """Enhanced TimeSeriesDataset that preserves original dataset indices for influence function computation."""
    
    def __init__(
        self,
        indexer,
        transform,
        sample_time_series=SampleTimeSeriesType.NONE,
        dataset_weight=1.0,
        global_offset=0,
    ):
        super().__init__(indexer, transform, sample_time_series, dataset_weight)
        self.global_offset = global_offset
    
    def set_global_offset(self, offset: int):
        """Set the global offset for this dataset within a concatenated dataset."""
        self.global_offset = offset
    
    def __getitem__(self, idx: int) -> dict[str, FlattenedData]:
        if idx < 0 or idx >= len(self):
            raise IndexError(
                f"Index {idx} out of range for dataset of length {len(self)}"
            )

        original_idx = idx  # Store the original index before any transformations
        
        if self.sample_time_series != SampleTimeSeriesType.NONE:
            # If sampling is used, we still want to track the original sampled index
            sampled_idx = np.random.choice(len(self.probabilities), p=self.probabilities)
            original_idx = sampled_idx  # Track the actual sampled index
            idx = sampled_idx

        # Get the data and add the GLOBAL dataset index metadata
        data = self._get_data(idx)
        # For weighted datasets, use the actual data index (bounded by num_ts) not the virtual index
        # This ensures global_idx never exceeds the allocated range based on num_ts
        actual_data_idx = idx % self.num_ts  # This matches what _get_data uses
        global_idx = self.global_offset + actual_data_idx
        if global_idx >= self.global_offset + self.num_ts:
            print(f"Global idx {global_idx} exceeds allocated range [{self.global_offset} - {self.global_offset + self.num_ts}]")
        data['_dataset_idx'] = global_idx
        # print(f"Original idx: {original_idx}, Actual data idx: {actual_data_idx}, Global offset: {self.global_offset}, Global idx: {global_idx}")
        return self.transform(self._flatten_data(data))


class MultiSampleTimeSeriesDataset(TimeSeriesDataset):
    def __init__(
        self,
        indexer: Indexer[dict[str, Any]],
        transform: Transformation,
        max_ts: int,
        combine_fields: tuple[str, ...],
        sample_time_series: SampleTimeSeriesType = SampleTimeSeriesType.NONE,
        dataset_weight: float = 1.0,
        sampler: Sampler = get_sampler("beta_binomial", a=2, b=5),
    ):
        super().__init__(indexer, transform, sample_time_series, dataset_weight)
        self.max_ts = max_ts
        self.combine_fields = combine_fields
        self.sampler = sampler

    def _get_data(self, idx: int) -> dict[str, BatchedData]:
        n_series = self.sampler(min(self.num_ts, self.max_ts))
        choices = np.concatenate([np.arange(idx), np.arange(idx + 1, self.num_ts)])
        others = np.random.choice(choices, n_series - 1, replace=False)
        samples = self.indexer[np.concatenate([[idx], others])]
        return samples

    def _flatten_data(
        self, samples: dict[str, BatchedData]
    ) -> dict[str, FlattenedData]:
        for field in samples.keys():
            if field in self.combine_fields:
                item = samples[field]
                if isinstance(item, list) and isinstance(item[0], MultivarTimeSeries):
                    samples[field] = [
                        univar for sample in samples[field] for univar in sample
                    ]
            elif isinstance(samples[field], BatchedDateTime):
                samples[field] = np.asarray(samples[field][0])
            elif isinstance(samples[field], BatchedString):
                samples[field] = samples[field][0]
            else:
                raise AssertionError(
                    f"Field {field} not accounted for in {self.indexer} MultiSampleTimeSeriesDataset"
                )
        return samples


class MultiSampleTimeSeriesDatasetWithIndex(MultiSampleTimeSeriesDataset):
    """Enhanced MultiSampleTimeSeriesDataset that preserves original dataset indices for influence function computation."""
    
    def __init__(
        self,
        indexer: Indexer[dict[str, Any]],
        transform: Transformation,
        max_ts: int,
        combine_fields: tuple[str, ...],
        sample_time_series: SampleTimeSeriesType = SampleTimeSeriesType.NONE,
        dataset_weight: float = 1.0,
        sampler: Sampler = get_sampler("beta_binomial", a=2, b=5),
        global_offset: int = 0,
    ):
        super().__init__(indexer, transform, max_ts, combine_fields, sample_time_series, dataset_weight, sampler)
        self.global_offset = global_offset
    
    def set_global_offset(self, offset: int):
        """Set the global offset for this dataset within a concatenated dataset."""
        self.global_offset = offset
    
    def __getitem__(self, idx: int) -> dict[str, FlattenedData]:
        if idx < 0 or idx >= len(self):
            raise IndexError(
                f"Index {idx} out of range for dataset of length {len(self)}"
            )

        original_idx = idx  # Store the original index before any transformations
        
        if self.sample_time_series != SampleTimeSeriesType.NONE:
            # If sampling is used, we still want to track the original sampled index
            sampled_idx = np.random.choice(len(self.probabilities), p=self.probabilities)
            original_idx = sampled_idx  # Track the actual sampled index
            idx = sampled_idx

        # Get the data and add the GLOBAL dataset index metadata
        data = self._get_data(idx)
        # For weighted datasets, use the actual data index (bounded by num_ts) not the virtual index
        # This ensures global_idx never exceeds the allocated range based on num_ts
        actual_data_idx = idx % self.num_ts  # This matches what _get_data uses
        global_idx = self.global_offset + actual_data_idx
        if global_idx >= self.global_offset + self.num_ts:
            print(f"Global idx {global_idx} exceeds allocated range [{self.global_offset} - {self.global_offset + self.num_ts}]")
        data['_dataset_idx'] = global_idx
        # print(f"Original idx: {original_idx}, Actual data idx: {actual_data_idx}, Global offset: {self.global_offset}, Global idx: {global_idx}")
        return self.transform(self._flatten_data(data))

    def _flatten_data(
        self, samples: dict[str, BatchedData]
    ) -> dict[str, FlattenedData]:
        """Override to handle the _dataset_idx field for influence function tracking."""
        # Handle the special _dataset_idx field
        dataset_idx = None
        if '_dataset_idx' in samples:
            dataset_idx = samples.pop('_dataset_idx')
        
        # Process the normal fields using the parent method
        flattened = super()._flatten_data(samples)
        
        # Add back the dataset index field
        if dataset_idx is not None:
            flattened['_dataset_idx'] = dataset_idx
            
        return flattened


class EvalDataset(TimeSeriesDataset):
    def __init__(
        self,
        windows: int,
        indexer: Indexer[dict[str, Any]],
        transform: Transformation,
        sample_time_series: SampleTimeSeriesType = SampleTimeSeriesType.NONE,
    ):
        super().__init__(
            indexer,
            transform,
            sample_time_series,
            dataset_weight=windows,
        )

    def _get_data(self, idx: int) -> dict[str, Data]:
        window, idx = divmod(idx, self.num_ts)
        item = self.indexer[idx]
        item["window"] = window
        return item


class EvalDatasetWithIndex(EvalDataset):
    """Enhanced EvalDataset that preserves original dataset indices for influence function computation."""
    
    def __init__(
        self,
        windows: int,
        indexer: Indexer[dict[str, Any]],
        transform: Transformation,
        sample_time_series: SampleTimeSeriesType = SampleTimeSeriesType.NONE,
        global_offset: int = 0,
    ):
        super().__init__(windows, indexer, transform, sample_time_series)
        self.global_offset = global_offset
    
    def set_global_offset(self, offset: int):
        """Set the global offset for this dataset within a concatenated dataset."""
        self.global_offset = offset
    
    def __getitem__(self, idx: int) -> dict[str, FlattenedData]:
        if idx < 0 or idx >= len(self):
            raise IndexError(
                f"Index {idx} out of range for dataset of length {len(self)}"
            )

        original_idx = idx  # Store the original index before any transformations
        
        if self.sample_time_series != SampleTimeSeriesType.NONE:
            # If sampling is used, we still want to track the original sampled index  
            sampled_idx = np.random.choice(len(self.probabilities), p=self.probabilities)
            original_idx = sampled_idx  # Track the actual sampled index
            idx = sampled_idx

        # Get the data and add the GLOBAL dataset index metadata
        data = self._get_data(idx)
        # For EvalDataset, use the actual time series index (from divmod) not the virtual window index
        # This ensures global_idx never exceeds the allocated range based on num_ts
        window, actual_ts_idx = divmod(idx, self.num_ts)
        global_idx = self.global_offset + actual_ts_idx
        data['_dataset_idx'] = global_idx
        # print(f"Eval - Original idx: {original_idx}, Window: {window}, Actual TS idx: {actual_ts_idx}, Global offset: {self.global_offset}, Global idx: {global_idx}")
        return self.transform(self._flatten_data(data))
