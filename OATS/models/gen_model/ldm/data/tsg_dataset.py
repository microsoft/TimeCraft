# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import numpy as np
from datetime import datetime
import pandas as pd
from distutils.util import strtobool
from statsmodels.distributions.empirical_distribution import ECDF
from torch.utils.data import WeightedRandomSampler
from einops import rearrange
        
class TSGDataset(Dataset):  # For generation task. Unified Univariate Generation Dataset
    def __init__(self, data_dict: dict):
        for key, data in data_dict.items():
            assert data.ndim == 3, f"Data must be 3D, but {key} got {data.ndim}D."
            assert data.shape[2] == 1, f"Only univariate time series are supported, but {key} got {data.shape[2]} channels."
        self.data_dict = data_dict
        self.cal_data_stats()
        
    def cal_data_stats(self):
        total_items = 0
        n_items_dict = {}
        key_list = []
        key_idx_list = []
        
        for key, data in self.data_dict.items():
            num_items = data.shape[0]
            total_items += num_items
            n_items_dict[key] = num_items
            key_list.append(key)
            key_idx_list.append(total_items)
            
        self.total_items = total_items
        self.items_dict = n_items_dict    
        self.key_list = key_list
        self.key_idx_list = np.array(key_idx_list)
    
    def get_reweight_sampler(self):
        dataset_weights = np.array([1 / self.items_dict[key] for key in self.key_list], dtype=np.float32)
        sample_weights = np.repeat(dataset_weights, [self.items_dict[key] for key in self.key_list])
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=self.total_items, replacement=True)
        return sampler
        
    def __len__(self):
        return self.total_items  # self.num_slices
    
    def __getitem__(self, idx):
        assert idx < self.total_items, f"Index({idx}) must be less than number of items({self.total_items})."
        data_key = np.where(self.key_idx_list > idx)[0].min()  # np.argmin(self.key_idx_list > idx)
        data_start_idx = self.key_idx_list[data_key-1] if data_key > 0 else 0
        data: np.ndarray = self.data_dict[self.key_list[data_key]]
        
        valid_idx = idx  - data_start_idx
        context = data[valid_idx,:,0]

        return {
            'context': context,  # shape: (window,)
            'data_key': data_key
            }  

    
class TSGDataModule(pl.LightningDataModule):
    '''
    Data module for unified time series generation task.
    Slicing is also done with this module. So the train/val is i.i.d within train dataset.
    '''
    def __init__(self, data_path_dict, window=96, val_portion=0.1, as_tensor:bool=True, normalize="centered_pit", batch_size=128, num_workers=0, pin_memory=True, drop_last=False, reweight=False, input_channels=1, **kwargs):
        super().__init__()
        self.data_path_dict = data_path_dict  # {data_name: data_path}
        self.data_dict = {}
        self.norm_data_dict = {}
        self.normalizer_dict = {}
        self.norm_train_dict = {}
        self.norm_val_dict = {}
        self.window = window
        self.val_portion = val_portion
        self.as_tensor = as_tensor
        assert normalize in [None, 'zscore', 'robust_iqr', 'robust_mad', 'pit', 'centered_pit', 'minmax'], f"Normalize({normalize}) must be in (zscore, robust_iqr, robust_mad, pit)."
        self.normalize = normalize
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.kwargs = kwargs
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.reweight = reweight
        self.input_channels = input_channels
        # self.transform = None
        self.kwargs = kwargs
        self.key_list = []
        self.drop_last = drop_last
        
    def prepare_data(self) -> None:
        print(f"Normalizing data with: {self.normalize}")
        self.key_list = []
        for data_name, data_path in self.data_path_dict.items():
            self.key_list.append(data_name)
            this_data = load_data_from_file(data_path).astype(np.float32)
            if this_data.ndim == 3:  # in shape (N, T, C)
                this_data = rearrange(this_data, 'n t c -> (n c) t 1')  # to shape (N*C, T)
            elif this_data.ndim == 2:
                this_data = this_data[..., np.newaxis]  # make shape (N, T, 1)
            else:
                raise ValueError(f"Unsupported data shape: {this_data.shape}")
            # first normalize, then split
            normalizer = self.fit_normalizer(this_data)
            self.data_dict[data_name] = this_data
            self.normalizer_dict[data_name] = normalizer
            norm_data = self.transform(this_data, normalizer)
            self.norm_data_dict[data_name] = norm_data
            train_data, val_data = self.split_train_val(norm_data)  # slice and split here
            self.norm_train_dict[data_name] = train_data
            self.norm_val_dict[data_name] = val_data
                        
            print(f"Loaded data: {data_name}; Train shape: {train_data.shape}, Validation shape: {val_data.shape}.")
            print(f"With normalizer fit as: {normalizer}")
    
    def split_train_val(self, data: np.ndarray):
        # By default, data are sliced into non-overlapped sequences.
        # shuffle stack_data, only along the first dimension
        np.random.shuffle(data)
        total_instances = data.shape[0]
        num_val_instances = int(total_instances * self.val_portion)
        train_data = data[:-num_val_instances]
        val_data = data[-num_val_instances:]
        
        return train_data, val_data
        
    def train_dataloader(self):
        train_dataset = TSGDataset(self.norm_train_dict)
        sampler = None
        if self.reweight:
            sampler = train_dataset.get_reweight_sampler()
            return DataLoader(train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory, drop_last=self.drop_last, sampler=sampler, **self.kwargs)
        else:
            return DataLoader(train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory, shuffle=True, drop_last=self.drop_last, **self.kwargs)
    
    def val_dataloader(self):
        val_dataset = TSGDataset(self.norm_val_dict)
        return DataLoader(val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory, **self.kwargs)
    
    def test_dataloader(self, **kwargs):
        return None
       
    def fit_normalizer(self, data: np.ndarray):
        normalizer = {}
        data = data.flatten()
        if self.normalize == 'zscore':
            normalizer['mean'] = np.nanmean(data)
            normalizer['std'] = np.nanstd(data)
        elif self.normalize == 'robust_iqr':
            normalizer['median'] = np.median(data)
            normalizer['iqr'] = np.subtract(*np.percentile(data, [75, 25]))
        elif self.normalize == 'robust_mad':
            normalizer['median'] = np.median(data)
            normalizer['mad'] = np.median(np.abs(data - normalizer['median']))
        elif self.normalize == 'minmax':
            normalizer['min'] = np.nanmin(data)
            normalizer['max'] = np.nanmax(data)
        elif self.normalize == 'pit' or self.normalize == 'centered_pit':
            ecdf = ECDF(data)
            normalizer['ecdf'] = ecdf
        return normalizer
    
    def transform(self, data: np.ndarray, normalizer=None, data_name=None):
        # if data_name is specified, the normalizer argument will be ignored.
        assert normalizer is not None or data_name is not None, "Must specify either normalizer or data name."
        if data_name is not None:
            assert data_name in self.normalizer_dict.keys(), f"Data name({data_name}) must be in normalizer dict key."
            normalizer = self.normalizer_dict[data_name]
        if self.normalize == 'zscore':
            return (data - normalizer['mean']) / (normalizer['std'] + 1e-8)
        elif self.normalize == 'robust_iqr':
            return (data - normalizer['median']) / (normalizer['iqr'] + 1e-8)
        elif self.normalize == 'robust_mad':
            return (data - normalizer['median']) / (normalizer['mad'] + 1e-8)
        if self.normalize == 'minmax':
            return (data - normalizer['min']) / (normalizer['max'] - normalizer['min'] + 1e-8)
        elif self.normalize == 'pit' or self.normalize == 'centered_pit':
            data_shape = data.shape
            norm_data = normalizer['ecdf'](data.flatten()).reshape(data_shape)
            if self.normalize == 'centered_pit':
                norm_data = norm_data * 2 - 1
            return norm_data
        
    def inverse_transform(self, data: np.ndarray, normalizer=None, data_name=None):
        # if data_name is specified, the normalizer argument will be ignored.
        assert normalizer is not None or data_name is not None, "Must specify either normalizer or data name."
        if data_name is not None:
            assert data_name in self.normalizer_dict.keys(), f"Data name({data_name}) must be in normalizer dict key."
            normalizer = self.normalizer_dict[data_name]
        if self.normalize == 'zscore':
            return data * normalizer['std'] + normalizer['mean']
        elif self.normalize == 'robust_iqr':
            return data * normalizer['iqr'] + normalizer['median']
        elif self.normalize == 'robust_mad':
            return data * normalizer['mad'] + normalizer['median']
        if self.normalize == 'minmax':
            return data * (normalizer['max'] - normalizer['min']) + normalizer['min']
        elif self.normalize == 'pit' or self.normalize == 'centered_pit':
            ecdf: ECDF = normalizer['ecdf']
            ecdf.x[0] = ecdf.x[1]
            if self.normalize == 'centered_pit':
                data = (data + 1) / 2
            return np.interp(data, ecdf.y, ecdf.x)

def load_data_from_file(file_path: str):
    if file_path.endswith(".csv"):
        loaded_data = pd.read_csv(file_path)
        return loaded_data.values  # no index columns, by default.
    elif file_path.endswith(".tsf"):
        loaded_data, frequency, forecast_horizon, contain_missing_values, contain_equal_length = convert_tsf_to_dataframe(
            file_path, 
            replace_missing_vals_with="NaN",
            value_column_name="series_value"
            )
        data = np.stack(loaded_data['series_value'].values).T
        return data  # no date column
    elif file_path.endswith(".npy"):
        loaded_data = np.load(file_path)  # shape like (N, T) by default
        return loaded_data.T  


def convert_tsf_to_dataframe(
    full_file_path_and_name,
    replace_missing_vals_with="NaN",
    value_column_name="series_value",
):
    col_names = []
    col_types = []
    all_data = {}
    line_count = 0
    frequency = None
    forecast_horizon = None
    contain_missing_values = None
    contain_equal_length = None
    found_data_tag = False
    found_data_section = False
    started_reading_data_section = False

    with open(full_file_path_and_name, "r", encoding="cp1252") as file:
        for line in file:
            # Strip white space from start/end of line
            line = line.strip()

            if line:
                if line.startswith("@"):  # Read meta-data
                    if not line.startswith("@data"):
                        line_content = line.split(" ")
                        if line.startswith("@attribute"):
                            if (
                                len(line_content) != 3
                            ):  # Attributes have both name and type
                                raise Exception("Invalid meta-data specification.")

                            col_names.append(line_content[1])
                            col_types.append(line_content[2])
                        else:
                            if (
                                len(line_content) != 2
                            ):  # Other meta-data have only values
                                raise Exception("Invalid meta-data specification.")

                            if line.startswith("@frequency"):
                                frequency = line_content[1]
                            elif line.startswith("@horizon"):
                                forecast_horizon = int(line_content[1])
                            elif line.startswith("@missing"):
                                contain_missing_values = bool(
                                    strtobool(line_content[1])
                                )
                            elif line.startswith("@equallength"):
                                contain_equal_length = bool(strtobool(line_content[1]))

                    else:
                        if len(col_names) == 0:
                            raise Exception(
                                "Missing attribute section. Attribute section must come before data."
                            )

                        found_data_tag = True
                elif not line.startswith("#"):
                    if len(col_names) == 0:
                        raise Exception(
                            "Missing attribute section. Attribute section must come before data."
                        )
                    elif not found_data_tag:
                        raise Exception("Missing @data tag.")
                    else:
                        if not started_reading_data_section:
                            started_reading_data_section = True
                            found_data_section = True
                            all_series = []

                            for col in col_names:
                                all_data[col] = []

                        full_info = line.split(":")

                        if len(full_info) != (len(col_names) + 1):
                            raise Exception("Missing attributes/values in series.")

                        series = full_info[len(full_info) - 1]
                        series = series.split(",")

                        if len(series) == 0:
                            raise Exception(
                                "A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series. Missing values should be indicated with ? symbol"
                            )

                        numeric_series = []

                        for val in series:
                            if val == "?":
                                numeric_series.append(replace_missing_vals_with)
                            else:
                                numeric_series.append(float(val))

                        if numeric_series.count(replace_missing_vals_with) == len(
                            numeric_series
                        ):
                            raise Exception(
                                "All series values are missing. A given series should contains a set of comma separated numeric values. At least one numeric value should be there in a series."
                            )

                        all_series.append(pd.Series(numeric_series).array)

                        for i in range(len(col_names)):
                            att_val = None
                            if col_types[i] == "numeric":
                                att_val = int(full_info[i])
                            elif col_types[i] == "string":
                                att_val = str(full_info[i])
                            elif col_types[i] == "date":
                                att_val = datetime.strptime(
                                    full_info[i], "%Y-%m-%d %H-%M-%S"
                                )
                            else:
                                raise Exception(
                                    "Invalid attribute type."
                                )  # Currently, the code supports only numeric, string and date types. Extend this as required.

                            if att_val is None:
                                raise Exception("Invalid attribute value.")
                            else:
                                all_data[col_names[i]].append(att_val)

                line_count = line_count + 1

        if line_count == 0:
            raise Exception("Empty file.")
        if len(col_names) == 0:
            raise Exception("Missing attribute section.")
        if not found_data_section:
            raise Exception("Missing series information under data section.")

        all_data[value_column_name] = all_series
        loaded_data = pd.DataFrame(all_data)

        return (
            loaded_data,
            frequency,
            forecast_horizon,
            contain_missing_values,
            contain_equal_length,
        )
