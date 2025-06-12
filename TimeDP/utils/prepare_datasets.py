# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
import numpy as np
import pandas as pd
from gluonts.dataset.repository.datasets import dataset_recipes, get_dataset
import requests
import zipfile
import os
import pandas as pd
import numpy as np
from einops import rearrange
from distutils.util import strtobool
from datetime import datetime
import os
import sys
from pathlib import Path
sys.path.append('..')
from utils.data_utils import convert_tsf_to_dataframe

PREFIX = './data/'
def download_monash_dataset():
    url_map = {
        'temprain': 'https://zenodo.org/records/5129091/files/temperature_rain_dataset_without_missing_values.zip?download=1',
        'wind_4_seconds': 'https://zenodo.org/records/4656032/files/wind_4_seconds_dataset.zip?download=1',
        'pedestrian': 'https://zenodo.org/records/4656626/files/pedestrian_counts_dataset.zip?download=1'
    }
    for dataset_name in ['temprain', 'wind_4_seconds', 'pedestrian']:
        # download the zip file
        url = url_map[dataset_name]
        zip_path = f"{dataset_name}.zip"

        print("Downloading dataset...")
        response = requests.get(url, stream=True)
        with open(zip_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
        print("Download complete.")

        # Unzip the ZIP file
        extract_path = "./data"
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_path)
        print(f"Files extracted to {extract_path}")

        # Deleting the ZIP file
        os.remove(zip_path)
        print("ZIP file deleted.")

def download_timegan_stock_dataset():
    url = 'https://raw.githubusercontent.com/jsyoon0823/TimeGAN/refs/heads/master/data/stock_data.csv'
    
    response = requests.get(url, stream=True)
    with open('stock_data.csv', "wb") as file:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                file.write(chunk)
    print("Download complete.")

def create_dataset_csv(dataset_simple,):
    print('--------------------------------------------------')
    print(f'create {dataset_simple} dataset csv')
    dataset_alias = { 
        'solar':'solar_nips',
        'electricity': 'electricity',
        'traffic':'traffic_nips',
        'kddcup':'kdd_cup_2018_without_missing',
        'taxi':'taxi_30min',
        'exchange':'exchange_rate_nips',
        'fred_md':'fred_md',
        'nn5': 'nn5_daily_without_missing',
        'web': 'kaggle_web_traffic_without_missing'
        }
    
    dataset_name = dataset_alias[dataset_simple]
    dataset = get_dataset(dataset_name, regenerate=False)
    metadata, train_data, test_data = dataset.metadata, dataset.train, dataset.test
    print("metadata", metadata)
    train_grouper = MultivariateGrouper(max_target_dim=min(2000, int(dataset.metadata.feat_static_cat[0].cardinality)))
    test_grouper = MultivariateGrouper(num_test_dates=int(len(dataset.test)/len(dataset.train)), 
                                    max_target_dim=min(2000, int(dataset.metadata.feat_static_cat[0].cardinality)))

    print("prepare the dataset")
    print(f'len(train_data): {len(train_data)}, len(test_data): {len(test_data)}')

    # group the dataset
    train_data=train_grouper(dataset.train)
    if dataset_simple == 'kddcup': # contains some special cases
        test_data = list(test_data)
        for i in range(len(test_data)):
            if len(test_data[i]['target']) == 10898:
                test_data[i]['target'] = np.concatenate((test_data[i]['target'], np.zeros(8)))
        test_data = test_grouper(test_data)
    else:
        test_data=test_grouper(dataset.test)

    # merge the train and test data
    train_data = list(train_data)
    test_data = list(test_data)
    print(f'train_data.shape: {train_data[0]["target"].shape}')
    print(f'test_data.shape: {test_data[-1]["target"].shape}')
    train_data_T = np.array(train_data[0]['target']).T
    test_data_T = np.array(test_data[-1]['target']).T
    print(f'train_data_T.shape: {train_data_T.shape}')
    print(f'test_data_T.shape: {test_data_T.shape}')

    print(f'train_data_T[-1][:10]: {train_data_T[-1][:10]}')
    print(f'test_data_T[-1][:10]: {test_data_T[-1][:10]}')


    prediction_length = metadata.prediction_length
    test_length = len(test_data)*prediction_length
    if dataset_simple =='taxi':
        # no train overlap 
        test_data_T_unic = test_data_T[-test_length-prediction_length:]
    else:
        # train overlap 
        test_data_T_unic = test_data_T[-test_length:]

    print((train_data_T[-1][-10:]))
    print((test_data_T[-len(test_data)*prediction_length-1][-10:]))
    
    data_all = np.concatenate((train_data_T, test_data_T_unic), axis=0)
    print(f'data_all.shape: {data_all.shape}')

    # generate dataframe
    metadata = dataset.metadata
    print("metadata", metadata)
    freq = metadata.freq

    start = pd.Timestamp("2012-01-01 00:00:00")  # Assume starting at 2012-01-01 00:00:00
    index = pd.date_range(start=start, freq=freq, periods=len(data_all))  # generate time series, interval is freq, length is len(data_all)
    df = pd.DataFrame(data_all, index=index, columns=range(data_all.shape[1]))  # create a dataframe, index is time series, columns is 0,1,2,3,4,5,6,7,8,9
    df.index.name = 'date'
    print(f'df.shape: {df.shape}')

    test_len = len(test_data)*prediction_length
    valid_len = min(7* prediction_length, test_len)
    train_len = len(df) - test_len - valid_len

    if dataset_simple == 'taxi':
        train_len = len(df) - test_len - valid_len - prediction_length  # exclude extra test data

    print("train_len", train_len)
    print("valid_len", valid_len)
    print("test_len", test_len)
    print("prediction_length", prediction_length)
    
    df.to_csv(f'./data/{dataset_simple}.csv', index=False)
    print(f'./data/{dataset_simple}.csv saved')

def more_data_loading(data_name, seq_len=168, stride=1, univar=False):
    if data_name in data_name_path_map:
            data_path = PREFIX + data_name_path_map[data_name]
            if data_name in ['stock']:
                ori_data = np.loadtxt(data_path, delimiter = ",",skiprows = 1)
            else:  # no index column
                ori_data = pd.read_csv(data_path).values
    elif data_name in monash_map:
        data_path = PREFIX + monash_map[data_name]
        loaded_data, *_ = convert_tsf_to_dataframe(data_path)
        ori_data = np.stack(loaded_data['series_value'].values).T
    # return ori_data
    temp_data = []    
    # Cut data by sequence length
    for i in range(0, len(ori_data) - seq_len + 1, stride):  # we do some slicing here
        _x = ori_data[None, i:i + seq_len]
        temp_data.append(_x)

    data = np.vstack(temp_data)
    if univar:
        data = rearrange(data, 'n t c -> (n c) t 1')
                
    return data # , ori_data

monash_map = {
    'temprain': './data/temperature_rain_dataset_without_missing_values.tsf',
    'wind_4_seconds': './data/wind_4_seconds_dataset.tsf',
    'pedestrian': './data/pedestrian_counts_dataset.tsf'
}
data_name_path_map = {
    'stock': 'stock_data.csv',
    'solar': './data/solar.csv',
    'electricity': './data/electricity.csv',
    'traffic': './data/traffic.csv',
    'kddcup': './data/kddcup.csv',
    'taxi': './data/taxi.csv',
    'exchange': './data/exchange.csv',
    'fred_md': './data/fred_md.csv',
    'nn5': './data/nn5.csv',
    'web': './data/web.csv',
    'temp': './data/temp.csv',
    'rain': './data/rain.csv',
    'pedestrian': './data/pedestrian.csv'
}

if __name__ == '__main__':
    download_monash_dataset()
    download_timegan_stock_dataset()
    transformed_dataset = ['solar','electricity','traffic','kddcup','taxi','exchange','fred_md','nn5','web']
    for dataset in transformed_dataset:
        try:
            create_dataset_csv(dataset)
        except:
            print(f'{dataset} failed')
            pass

    data_path = PREFIX + monash_map['temprain']
    loaded_data, *_ = convert_tsf_to_dataframe(data_path)
    loaded_data.head()

    rain = np.stack(loaded_data.loc[loaded_data['obs_or_fcst'] == 'PRCP_SUM']['series_value'].values)
    print(rain.shape)
    df = pd.DataFrame(rain.T)
    df.to_csv(PREFIX + 'rain.csv', index=False)

    temp = np.stack(loaded_data.loc[loaded_data['obs_or_fcst'] == 'T_MEAN']['series_value'].values)
    print(temp.shape)
    df = pd.DataFrame(temp.T)
    df.to_csv(PREFIX + 'temp.csv', index=False)
    
    mix_dataset = [
            'solar', 'electricity', 'traffic', 'kddcup', 'taxi', 'exchange', 'fred_md', 'nn5', 'temp', 'rain', 'wind_4_seconds'
        ]
    for seq_len in [24, 96, 168, 336]:
        stride = seq_len
        for data_name in mix_dataset:
            ori_data = more_data_loading(data_name, seq_len, stride)
            print(data_name, ori_data.shape)
            test_portion = max(1, int(ori_data.shape[0] * 0.1))
            train_data = ori_data[:-test_portion]
            val_data = ori_data[-test_portion:]

            np.save(PREFIX + f'{data_name}_{seq_len}_train.npy', train_data)
            np.save(PREFIX + f'{data_name}_{seq_len}_val.npy', val_data)
            print(train_data.shape, val_data.shape)
            
        # pedestrian contains inconsistent length
        data_path = PREFIX + monash_map['pedestrian']
        loaded_data, *_ = convert_tsf_to_dataframe(data_path)
        loaded_data.head()
        
        train_part = []
        val_part = []
        for _, x in loaded_data['series_value'].items():
            num_segments = x.shape[0] // stride
            val_num_segments = max(num_segments // 10, 1)
            all_segments = []
            for i in range(0, x.shape[0] - seq_len + 1, stride):  # we do some slicing here, but not now.
                _x = x[None, i:i + seq_len]
                all_segments.append(_x)
            train_part += all_segments[:-val_num_segments]
            val_part += all_segments[-val_num_segments:]
            print(num_segments, val_num_segments)

        pedestrian_train = np.vstack(train_part)
        pedestrian_val = np.vstack(val_part)
        print(pedestrian_train.shape, pedestrian_val.shape)

        np.save(PREFIX + f'pedestrian_{seq_len}_train.npy', pedestrian_train[:,:,None])
        np.save(PREFIX + f'pedestrian_{seq_len}_val.npy', pedestrian_val[:,:,None])

    zero_shot_schedule = [3, 10, 100]
    for data_name in ['web', 'stock']:  # 'web', 'stock'
        for seq_len in [24, 96, 168, 336]:
            if data_name == 'stock':
                ori_data = more_data_loading(data_name, seq_len, 1, univar=False)
                uni_ori_data = ori_data[:,:,0,None]
                uni_ori_data /= uni_ori_data[:, :1, :]
            else:
                ori_data = more_data_loading(data_name, seq_len, seq_len, univar=False)
                uni_ori_data = rearrange(ori_data, 'b t c -> (b c) t 1')  # uniori_data[:,:,0,None]# 

            zero_shot_data_path = Path(f'{PREFIX}/ts_data/new_zero_shot_data')
            zero_shot_data_path.mkdir(exist_ok=True, parents=True)

            print(len(uni_ori_data))
            np.random.seed(0)
            k_idx = np.random.choice(len(uni_ori_data), 2000+max(zero_shot_schedule))
            zero_shot_test_data = uni_ori_data[k_idx[-2000:]]
            np.save(zero_shot_data_path/f'{data_name}_{seq_len}_test_sample.npy', zero_shot_test_data)
            for k in zero_shot_schedule:
                zero_shot_prompt = uni_ori_data[k_idx[:k]]
                np.save(zero_shot_data_path/f'{data_name}_{seq_len}_k_{k}_sample.npy', zero_shot_prompt)
                
                pd.DataFrame(zero_shot_prompt[:,:,0].T).to_csv(zero_shot_data_path/f'{data_name}_dim0_{seq_len}_k_{k}_sample.csv', index=False)
