# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
from typing import Any, Callable

from torch.utils.data import ConcatDataset, Dataset

from tsfm.transform import Transformation

import logging
log = logging.getLogger(__name__)

# TODO: Add __repr__
class DatasetBuilder(abc.ABC):
    @abc.abstractmethod
    def build_dataset(self, *args, **kwargs): ...

    @abc.abstractmethod
    def load_dataset(
        self, transform_map: dict[Any, Callable[..., Transformation]]
    ) -> Dataset: ...


class ConcatDatasetBuilder(DatasetBuilder):
    def __init__(self, *builders: DatasetBuilder):
        super().__init__()
        assert len(builders) > 0, "Must provide at least one builder to ConcatBuilder"
        assert all(
            isinstance(builder, DatasetBuilder) for builder in builders
        ), "All builders must be instances of DatasetBuilder"
        self.builders: tuple[DatasetBuilder, ...] = builders

    def build_dataset(self):
        raise ValueError(
            "Do not use ConcatBuilder to build datasets, build sub datasets individually instead."
        )

    def load_dataset(
        self, transform_map: dict[Any, Callable[..., Transformation]]
    ) -> ConcatDataset:
        return ConcatDataset(
            [builder.load_dataset(transform_map) for builder in self.builders]
        )


class ConcatDatasetBuilderWithGlobalIndex(DatasetBuilder):
    """ConcatDatasetBuilder that calculates and passes global offsets to sub-datasets."""
    
    def __init__(self, *builders: DatasetBuilder):
        super().__init__()
        assert len(builders) > 0, "Must provide at least one builder to ConcatBuilder"
        assert all(
            isinstance(builder, DatasetBuilder) for builder in builders
        ), "All builders must be instances of DatasetBuilder"
        self.builders: tuple[DatasetBuilder, ...] = builders
        self.dataset_ranges = []  # Store list of (start, end, name, builder_type) tuples

    def build_dataset(self):
        raise ValueError(
            "Do not use ConcatBuilder to build datasets, build sub datasets individually instead."
        )

    def _get_dataset_name(self, builder, dataset):
        """Extract a meaningful dataset name from builder and dataset."""
        # Method 1: Try to get the dataset name from dataset.info if it exists
        dataset_info_name = None
        if hasattr(dataset, 'info') and hasattr(dataset.info, 'dataset_name'):
            dataset_info_name = dataset.info.dataset_name
        
        # Method 2: Try to get from dataset indexer
        indexer_name = None
        if hasattr(dataset, 'indexer') and hasattr(dataset.indexer, 'dataset'):
            if hasattr(dataset.indexer.dataset, 'info') and hasattr(dataset.indexer.dataset.info, 'dataset_name'):
                indexer_name = dataset.indexer.dataset.info.dataset_name
        
        # Method 3: Look for dataset names in builder's dataset_list
        builder_dataset_name = None
        if hasattr(builder, 'dataset_list') and hasattr(builder, 'datasets'):
            # For LOTSA builders, try to match the dataset path or find in datasets list
            if hasattr(dataset, 'indexer') and hasattr(dataset.indexer, 'dataset_path'):
                dataset_path = str(dataset.indexer.dataset_path)
                for dataset_name in builder.dataset_list:
                    if dataset_name in dataset_path:
                        builder_dataset_name = dataset_name
                        break
            elif len(builder.dataset_list) == 1:
                # If builder only has one dataset, use that name
                builder_dataset_name = builder.dataset_list[0]
        
        # Method 4: Check if we can extract from file paths or other attributes
        path_name = None
        if hasattr(dataset, 'indexer') and hasattr(dataset.indexer, 'dataset'):
            if hasattr(dataset.indexer.dataset, '_data_dir_cache') and dataset.indexer.dataset._data_dir_cache:
                # Extract name from data directory path
                import os
                path_name = os.path.basename(dataset.indexer.dataset._data_dir_cache.rstrip('/'))
        
        # Prioritize the most specific and meaningful name
        best_name = None
        
        # First preference: builder dataset name (most specific)
        if builder_dataset_name and builder_dataset_name != 'generator':
            best_name = builder_dataset_name
        # Second preference: path name (usually specific)
        elif path_name and path_name != 'generator':
            best_name = path_name
        # Third preference: indexer name (if not generic)
        elif indexer_name and indexer_name != 'generator':
            best_name = indexer_name
        # Fourth preference: dataset info name (even if generic)
        elif dataset_info_name:
            best_name = dataset_info_name
        
        # If we still have a generic name, try to make it more specific using builder info
        if best_name == 'generator' or best_name is None:
            builder_name = builder.__class__.__name__.replace('DatasetBuilder', '')
            
            # For builders with multiple datasets, try to get more specific info
            if hasattr(builder, 'datasets') and builder.datasets:
                # If the builder is loading specific datasets, try to match
                if len(builder.datasets) == 1:
                    best_name = builder.datasets[0]
                else:
                    # Multiple datasets - we need to figure out which one this is
                    # This is tricky without more context, but we can use the builder name
                    best_name = f"{builder_name}_dataset"
            else:
                best_name = builder_name
        
        return best_name or "Unknown"

    def _flatten_datasets(self, dataset, global_offset, builder=None, dataset_name=None):
        """Recursively flatten nested ConcatDatasets and set global offsets."""
        flattened = []
        current_offset = global_offset
        
        if isinstance(dataset, ConcatDataset):
            for i, sub_dataset in enumerate(dataset.datasets):
                sub_flattened, current_offset = self._flatten_datasets(
                    sub_dataset, current_offset, builder, f"{dataset_name}_sub{i}" if dataset_name else None
                )
                flattened.extend(sub_flattened)
        else:
            # Single dataset - set its global offset and record metadata
            if hasattr(dataset, 'set_global_offset'):
                dataset.set_global_offset(current_offset)
                
            # Extract meaningful dataset name
            if dataset_name is None:
                dataset_name = self._get_dataset_name(builder, dataset)
            
            # Store range metadata for this dataset
            # Use num_ts (actual time series count) instead of len(dataset) (weighted count)
            # because the dataset classes use modulo num_ts for global_idx calculation
            if hasattr(dataset, 'num_ts'):
                actual_size = dataset.num_ts
            else:
                actual_size = len(dataset)  # fallback for datasets without num_ts
                
            self.dataset_ranges.append({
                'global_start': current_offset,
                'global_end': current_offset + actual_size - 1,
                'size': actual_size,
                'dataset_name': dataset_name,
                'builder_type': builder.__class__.__name__ if builder else 'Unknown'
            })
            
            log.info(f"Set global offset {current_offset} for dataset '{dataset_name}' with {actual_size} samples (range: {current_offset}-{current_offset + actual_size - 1})")
            log.info(f"Dataset ts_num: {getattr(dataset, 'num_ts', 'N/A')}.")
            flattened.append(dataset)
            current_offset += actual_size
        
        return flattened, current_offset

    def load_dataset(
        self, transform_map: dict[Any, Callable[..., Transformation]]
    ) -> ConcatDataset:
        all_datasets = []
        global_offset = 0
        self.dataset_ranges = []  # Reset ranges
        
        for builder in self.builders:
            # Load the dataset
            dataset = builder.load_dataset(transform_map)
            
            # Flatten and set offsets
            flattened, global_offset = self._flatten_datasets(dataset, global_offset, builder)
            all_datasets.extend(flattened)
        
        log.info(f"Total concatenated dataset size: {global_offset}")
        log.info(f"Dataset ranges: {len(self.dataset_ranges)} datasets")
        return ConcatDataset(all_datasets)

    def get_dataset_name_for_global_index(self, global_idx: int) -> str:
        """Get the sub-dataset name for a given global index."""
        # Bounds check: ensure global_idx is not negative
        if global_idx < 0:
            return f"Invalid_idx_{global_idx}"
        
        # Find the dataset that contains this global index
        for range_info in self.dataset_ranges:
            if range_info['global_start'] <= global_idx <= range_info['global_end']:
                return range_info['dataset_name']
        
        # If we reach here, the index is out of bounds
        # Check if it's beyond the maximum known range
        if self.dataset_ranges:
            max_range = max(r['global_end'] for r in self.dataset_ranges)
            total_size = max_range + 1
            
            if global_idx > max_range:
                log.warning(f"Global index {global_idx} exceeds dataset bounds (max: {max_range}, total: {total_size}). "
                          f"This might indicate stale influence scores from a previous run with larger dataset.")
                return f"OutOfBounds_idx_{global_idx}_max_{max_range}"
        
        return f"Unknown_idx_{global_idx}"
    
    def get_all_dataset_ranges(self) -> list:
        """Get all dataset range metadata for debugging/analysis."""
        return self.dataset_ranges.copy()


class SafeConcatDatasetBuilder(ConcatDatasetBuilder):
    def __init__(self, *builders: DatasetBuilder):
        super().__init__(*builders)
        self.builders = builders

    def load_dataset(
            self, transform_map: dict[Any, Callable[..., Transformation]]
        ) -> ConcatDataset:
            datasets = []
            for builder in self.builders:
                try:
                    datasets.append(builder.safeload_dataset(transform_map))
                except Exception as e:
                    log.error(f"Error loading dataset from {builder}: {e}")
                    continue
                
            return ConcatDataset(datasets)
    
