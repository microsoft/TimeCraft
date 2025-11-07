#  Copyright (c), Salesforce 2024,  
#  SPDX-License-Identifier: Apache-2
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# Modifications Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
import datasets
from collections.abc import Callable
from pathlib import Path
from typing import Optional, Any

from datasets import load_from_disk
from torch.utils.data import ConcatDataset, Dataset

from tsfm.common.core import abstract_class_property
from tsfm.common.env import env
from tsfm.data.builder._base import DatasetBuilder
from tsfm.data.dataset import SampleTimeSeriesType, TimeSeriesDataset, TimeSeriesDatasetWithIndex
from tsfm.data.indexer import HuggingFaceDatasetIndexer
from tsfm.transform import Identity, Transformation

import logging
log = logging.getLogger(__name__)

@abstract_class_property("dataset_list", "dataset_type_map", "dataset_load_func_map")
class LOTSADatasetBuilder(DatasetBuilder, abc.ABC):
    dataset_list: list[str] = NotImplemented
    dataset_type_map: dict[str, type[TimeSeriesDataset]] = NotImplemented
    dataset_load_func_map: dict[str, Callable[..., TimeSeriesDataset]] = NotImplemented
    uniform: bool = False
    
    # Use the enhanced dataset with index tracking by default
    default_dataset_class: type[TimeSeriesDataset] = TimeSeriesDatasetWithIndex

    def __init__(
        self,
        datasets: list[str],
        weight_map: Optional[dict[str, float]] = None,
        sample_time_series: SampleTimeSeriesType = SampleTimeSeriesType.NONE,
        storage_path: Path = env.LOTSA_V1_PATH,
    ):
        assert all(
            dataset in self.dataset_list for dataset in datasets
        ), f"Invalid datasets {set(datasets).difference(self.dataset_list)}, must be one of {self.dataset_list}"
        weight_map = weight_map or dict()
        self.datasets = datasets
        self.weights = [weight_map.get(dataset, 1.0) for dataset in datasets]
        self.sample_time_series = sample_time_series
        self.storage_path = storage_path

    def load_dataset(
        self, transform_map: dict[str | type, Callable[..., Transformation]]
    ) -> Dataset:
        datasets = [
            self.dataset_load_func_map[dataset](
                HuggingFaceDatasetIndexer(
                    add_column(
                        load_from_disk(str(self.storage_path / dataset)),
                        'dataset_name', 
                        dataset),
                    uniform=self.uniform,
                ),
                self._get_transform(transform_map, dataset),
                sample_time_series=self.sample_time_series,
                dataset_weight=weight,
            )
            for dataset, weight in zip(self.datasets, self.weights)
        ]

        return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    
    def safeload_dataset(
        self, transform_map: dict[str | type, Callable[..., Transformation]]
    ) -> Dataset:
        datasets = []
        for dataset, weight in zip(self.datasets, self.weights):
            try:
                indexer = HuggingFaceDatasetIndexer(
                    load_from_disk(str(self.storage_path / dataset)),
                    uniform=self.uniform,
                )
                datasets.append(
                    self.dataset_load_func_map[dataset](
                        indexer,
                        self._get_transform(transform_map, dataset),
                        sample_time_series=self.sample_time_series,
                        dataset_weight=weight,
                    )
                )
            except Exception as e:
                log.error(f'Error loading dataset from {dataset}: {e}')
                continue
        
        return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)

    def _get_transform(
        self,
        transform_map: dict[str | type, Callable[..., Transformation]],
        dataset: str,
    ) -> Transformation:

        if dataset in transform_map:
            transform = transform_map[dataset]
        elif (dataset_type := self.dataset_type_map[dataset]) in transform_map:
            transform = transform_map[dataset_type]
        else:
            try:  # defaultdict
                transform = transform_map[dataset]
            except KeyError:
                transform = transform_map.get("default", Identity)
        return transform()

def add_column(hfdataset: datasets.arrow_dataset.Dataset, column_name: str, clolumn_value: Any):
    new_clolumn = [clolumn_value] * len(hfdataset)
    hfdataset = hfdataset.add_column(column_name, new_clolumn)
    return hfdataset