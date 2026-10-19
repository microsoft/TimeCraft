
import warnings
warnings.filterwarnings("ignore")

import os
import sys
import argparse
import numpy as np
import torch


class Model_Basic(object):
    def __init__(self, args):
        from models import TimeXer, CATS, ModernTCN, PatchTST, PDF, TimeMixer, CARD, DLinear, LSINet, SEMixer, MLF

        self.args = args
        self.model_dict = {
            'TimeXer': TimeXer,
            'CATS': CATS, 'ModernTCN': ModernTCN, 'PatchTST': PatchTST,
            'PDF': PDF, 'TimeMixer': TimeMixer, 'CARD': CARD,
            'DLinear': DLinear, 'LSINet': LSINet, 'SEMixer': SEMixer,
            'MLF': MLF,
        }

    def bulid_model(self):
        self.device = torch.device(f'cuda:{self.args.gpu}') if self.args.use_gpu else torch.device('cpu')
        model = self.model_dict[self.args.model].Model(self.args).to(self.device)
        return model


ALL_MODELS = ['SEMixer', 'CARD', 'CATS', 'DLinear', 'LSINet',  'ModernTCN', 'MLF','PDF', 'PatchTST', 'TimeMixer', 'TimeXer']
ALL_DATASETS = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'exchange_rate', 'weather', 'electricity', 'traffic']
BASE_PATH = '/mnt/storage/checkpoints_forecast_RaFT_original_save_MTS'
SEED_NAME = 'random_seed_0'
SEQ_LEN = 96
PRED_LEN = 96


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    return False


def setup_base_args(gpu=0):
    """Use the same argparse as construct_dataset_merge_prelen_alldatasets.py to get all defaults"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_only', type=bool, default=False)
    parser.add_argument('--root_path', type=str, default='./data/ETT/')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv')
    parser.add_argument('--features', type=str, default='M')
    parser.add_argument('--target', type=str, default='OT')
    parser.add_argument('--freq', type=str, default='h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--label_len', type=int, default=48)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--fc_dropout', type=float, default=0.05)
    parser.add_argument('--head_dropout', type=float, default=0.0)
    parser.add_argument('--patch_len', type=int, default=16)
    parser.add_argument('--stride', type=int, default=8)
    parser.add_argument('--padding_patch', default='end')
    parser.add_argument('--revin', type=int, default=1)
    parser.add_argument('--affine', type=int, default=0)
    parser.add_argument('--subtract_last', type=int, default=0)
    parser.add_argument('--decomposition', type=int, default=0)
    parser.add_argument('--kernel_size', type=int, default=25)
    parser.add_argument('--individual', type=int, default=0)
    parser.add_argument('--embed_type', type=int, default=0)
    parser.add_argument('--enc_in', type=int, default=7)
    parser.add_argument('--dec_in', type=int, default=7)
    parser.add_argument('--c_out', type=int, default=7)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--e_layers', type=int, default=2)
    parser.add_argument('--d_layers', type=int, default=1)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--moving_avg', type=int, default=25)
    parser.add_argument('--factor', type=int, default=1)
    parser.add_argument('--distil', action='store_false', default=True)
    parser.add_argument('--dropout', type=float, default=0.05)
    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--activation', type=str, default='gelu')
    parser.add_argument('--output_attention', action='store_true', default=False)
    parser.add_argument('--do_predict', action='store_true', default=False)
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--itr', type=int, default=2)
    parser.add_argument('--train_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--patience', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--des', type=str, default='test')
    parser.add_argument('--loss', type=str, default='mse')
    parser.add_argument('--lradj', type=str, default='type3')
    parser.add_argument('--pct_start', type=float, default=0.3)
    parser.add_argument('--use_amp', action='store_true', default=False)
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--use_multi_gpu', action='store_true', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3')
    parser.add_argument('--test_flop', action='store_true', default=False)
    parser.add_argument('--num_experts_list', type=list, default=[4, 4, 4])
    parser.add_argument('--channel_independence', type=int, default=1)
    parser.add_argument('--decomp_method', type=str, default='moving_avg')
    parser.add_argument('--use_future_temporal_feature', type=int, default=0)
    parser.add_argument('--use_norm', type=int, default=1)
    parser.add_argument('--down_sampling_layers', type=int, default=0)
    parser.add_argument('--down_sampling_window', type=int, default=1)
    parser.add_argument('--down_sampling_method', type=str, default='avg')
    parser.add_argument('--modes1', type=int, default=64)
    parser.add_argument('--mode_type', type=int, default=0)
    parser.add_argument('--bucket_size', type=int, default=4)
    parser.add_argument('--n_hashes', type=int, default=4)
    parser.add_argument('--film_ours', default=True, action='store_true')
    parser.add_argument('--ab', type=int, default=2)
    parser.add_argument('--ratio', type=float, default=0.5)
    parser.add_argument('--film_version', type=int, default=0)
    parser.add_argument('--add', action='store_true', default=False)
    parser.add_argument('--wo_conv', action='store_true', default=False)
    parser.add_argument('--serial_conv', action='store_true', default=False)
    parser.add_argument('--kernel_list', type=int, nargs='+', default=[3, 7, 9])
    parser.add_argument('--period', type=int, nargs='+', default=[24, 12])
    parser.add_argument('--class_strategy', type=str, default='projection')
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--dp_rank', type=int, default=8)
    parser.add_argument('--rescale', type=int, default=1)
    parser.add_argument('--merge_size', type=int, default=2)
    parser.add_argument('--momentum', type=float, default=0.1)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--devices_number', type=int, default=1)
    parser.add_argument('--use_statistic', action='store_true', default=False)
    parser.add_argument('--use_decomp', action='store_true', default=False)
    parser.add_argument('--same_smoothing', action='store_true', default=False)
    parser.add_argument('--top_k', type=int, default=3)
    parser.add_argument('--num_kernels', type=int, default=6)
    parser.add_argument('--channel_mixing', type=int, default=1)
    parser.add_argument('--query_independence', action='store_true', default=False)
    parser.add_argument('--store_attn', action='store_true', default=False)
    parser.add_argument('--QAM_start', type=float, default=0.1)
    parser.add_argument('--revin_affine', type=int, default=0)
    parser.add_argument('--revin_subtract_last', type=int, default=0)
    parser.add_argument('--stem_ratio', type=int, default=8)
    parser.add_argument('--down_ratio', type=int, default=2)
    parser.add_argument('--fmap_size', type=int, default=768)
    parser.add_argument('--dims', nargs='+', type=int, default=[64, 128, 256, 512])
    parser.add_argument('--depths', nargs='+', type=int, default=[1, 1, 1, 1])
    parser.add_argument('--drop_path_rate', type=float, default=0.3)
    parser.add_argument('--layer_scale_value', nargs='+', type=float, default=[-1, -1, -1, -1])
    parser.add_argument('--use_pe', nargs='+', type=int, default=[1, 1, 1, 1])
    parser.add_argument('--use_lpu', nargs='+', type=int, default=[1, 1, 1, 1])
    parser.add_argument('--local_kernel_size', nargs='+', type=int, default=[3, 3, 3, 3])
    parser.add_argument('--expansion', type=int, default=4)
    parser.add_argument('--drop', type=float, default=0.0)
    parser.add_argument('--use_dwc_mlp', nargs='+', type=int, default=[1, 1, 1, 1])
    parser.add_argument('--heads', nargs='+', type=int, default=[4, 8, 16, 32])
    parser.add_argument('--attn_drop', type=float, default=0.0)
    parser.add_argument('--proj_drop', type=float, default=0.0)
    parser.add_argument('--stage_spec', nargs='+', type=list, default=[['D'], ['D'], ['D', 'D', 'D'], ['D']])
    parser.add_argument('--window_size', nargs='+', type=int, default=[3, 3, 3, 3])
    parser.add_argument('--nat_ksize', nargs='+', type=int, default=[3, 3, 3, 3])
    parser.add_argument('--ksize', nargs='+', type=int, default=[9, 7, 5, 3])
    parser.add_argument('--n_groups', nargs='+', type=int, default=[2, 4, 8, 16])
    parser.add_argument('--offset_range_factor', nargs='+', type=float, default=[-1, -1, -1, -1])
    parser.add_argument('--no_off', nargs='+', type=int, default=[0, 0, 0, 0])
    parser.add_argument('--dwc_pe', nargs='+', type=int, default=[0, 0, 0, 0])
    parser.add_argument('--fixed_pe', nargs='+', type=int, default=[0, 0, 0, 0])
    parser.add_argument('--log_cpb', nargs='+', type=int, default=[0, 0, 0, 0])
    parser.add_argument('--head_type', type=str, default='Flatten')
    parser.add_argument('--use_head_norm', type=int, default=1)
    parser.add_argument('--downsample_ratio', type=int, default=2)
    parser.add_argument('--ffn_ratio', type=int, default=2)
    parser.add_argument('--patch_size', type=int, default=8)
    parser.add_argument('--patch_stride', type=int, default=4)
    parser.add_argument('--num_blocks', nargs='+', type=int, default=[1, 1, 1, 1])
    parser.add_argument('--large_size', nargs='+', type=int, default=[31, 29, 27, 13])
    parser.add_argument('--small_size', nargs='+', type=int, default=[5, 5, 5, 5])
    parser.add_argument('--dw_dims', nargs='+', type=int, default=[256, 256, 256, 256])
    parser.add_argument('--small_kernel_merged', type=str2bool, default=False)
    parser.add_argument('--call_structural_reparam', type=bool, default=False)
    parser.add_argument('--use_multi_scale', type=str2bool, default=True)
    parser.add_argument('--conv_kernel', type=int, nargs='+', default=[17, 49])
    parser.add_argument('--decomp_kernel', type=int, nargs='+', default=[17, 49])
    parser.add_argument('--isometric_kernel', type=int, nargs='+', default=[17, 49])
    parser.add_argument('--use_ims', action='store_true', default=False)
    parser.add_argument('--use_weight_decay', type=int, default=0)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--decay_fac', type=float, default=0.75)

    args = parser.parse_args([])

    # Common overrides for MTS inference
    args.gpu = gpu
    args.device = f'cuda:{gpu}'
    args.use_gpu = True
    args.task_name = 'forecast'
    args.seq_len = SEQ_LEN
    args.pred_len = PRED_LEN
    args.seq_len_ori = SEQ_LEN
    args.pred_len_ori = PRED_LEN
    args.original = True
    args.itr = 1
    args.is_finetuning = True
    args.rand_noise = False
    args.output_len_list = None
    args.linear_prob = False
    args.linear_prob_random = False
    args.random_w = False
    args.re_train = False
    args.prtrain_w = False
    args.record = True
    args.test = True
    args.seed = 0
    args.use_multi_gpu = False
    args.num_workers = 4
    args.top_k = 5
    args.revin_norm = True
    args.D_norm = False
    return args


def set_dataset_model_args(args, data_name, model_name):
    """
    Copied directly from construct_dataset_merge_prelen_alldatasets.py lines 1579-2649.
    Sets dataset base args, then per-dataset per-model overrides, then global model overrides.
    Finally overrides enc_in=1 for MTS-as-univariate storage.
    """
    args.model = model_name
    args.data_type = data_name
    pre_len = PRED_LEN

    # ======================== Dataset base args ========================
    if data_name == 'ETTh1':
        args.data_path = 'ETTh1.csv'
        args.data = 'ETTh1'
        args.model_id = 'ETTh1'
        args.enc_in = 7
        args.d_model = 128
        args.d_ff = 128
        args.dropout = 0.3
        args.fc_dropout = 0.3
        args.head_dropout = 0
        args.batch_size = 128
        args.learning_rate = 0.0001
        args.e_layers = 3
        args.n_heads = 4
        if model_name == 'TimeMixer':
            args.down_sampling_layers = 3
            args.down_sampling_window = 2
            args.d_model = 16
            args.d_ff = 32
        elif model_name == 'PatchTST':
            args.d_model = 16
            args.d_ff = 128
        elif model_name == 'PDF':
            args.n_heads = 4
            args.d_model = 16
            args.d_ff = 128
            args.period = [24]
            args.patch_len = [1]
            args.stride = [1]
            args.kernel_list = [3, 7, 11]
        elif model_name == 'CARD':
            args.e_layers = 2
            args.n_heads = 2
            args.d_model = 16
            args.d_ff = 32
            args.dropout = 0.3
            args.fc_dropout = 0.3
            args.head_dropout = 0
            args.patch_len = 16
            args.stride = 8
        elif model_name == 'CATS':
            args.d_model = 256
            args.d_ff = 256
            args.n_heads = 16
            args.QAM_end = 0.2
            args.patch_len = 24
            args.stride = 24
            args.d_layers = 3

    elif data_name == 'ETTh2':
        args.data_path = 'ETTh2.csv'
        args.data = 'ETTh2'
        args.model_id = 'ETTh2'
        args.enc_in = 7
        args.d_model = 128
        args.d_ff = 128
        args.dropout = 0.3
        args.fc_dropout = 0.3
        args.head_dropout = 0
        args.batch_size = 128
        args.learning_rate = 0.0001
        args.e_layers = 3
        args.n_heads = 4
        if model_name == 'TimeMixer':
            args.down_sampling_layers = 3
            args.down_sampling_window = 2
            args.d_model = 16
            args.d_ff = 32
        elif model_name == 'PatchTST':
            args.d_model = 16
            args.d_ff = 128
        elif model_name == 'PDF':
            args.n_heads = 4
            args.d_model = 16
            args.d_ff = 128
            args.period = [24, 180, 720]
            args.patch_len = [4, 16, 48]
            args.stride = [4, 16, 48]
            args.kernel_list = [3, 7, 11]
        elif model_name == 'CARD':
            args.e_layers = 2
            args.n_heads = 2
            args.d_model = 16
            args.d_ff = 32
            args.dropout = 0.3
            args.fc_dropout = 0.3
            args.head_dropout = 0
            args.patch_len = 16
            args.stride = 8
        elif model_name == 'CATS':
            args.d_model = 256
            args.d_ff = 512
            args.n_heads = 16
            args.QAM_end = 0.2
            args.patch_len = 24
            args.stride = 24
            args.query_independence = True
            args.d_layers = 3

    elif data_name == 'ETTm1':
        args.data_path = 'ETTm1.csv'
        args.data = 'ETTm1'
        args.model_id = 'ETTm1'
        args.enc_in = 7
        args.n_heads = 16
        args.d_model = 128
        args.d_ff = 256
        args.dropout = 0.2
        args.fc_dropout = 0.2
        args.head_dropout = 0
        args.batch_size = 128
        args.learning_rate = 0.0001
        args.e_layers = 3
        if model_name == 'TimeMixer':
            args.down_sampling_layers = 3
            args.down_sampling_window = 2
            args.d_model = 16
            args.d_ff = 32
        elif model_name == 'PatchTST':
            args.d_model = 128
            args.d_ff = 256
        elif model_name == 'PDF':
            args.n_heads = 16
            args.d_model = 48
            args.d_ff = 48
            args.period = [48, 90, 102, 360, 720]
            args.patch_len = [3, 6, 16, 32, 48]
            args.stride = [3, 6, 16, 32, 48]
            args.kernel_list = [5, 7, 11, 15]
        elif model_name == 'CARD':
            args.e_layers = 2
            args.n_heads = 2
            args.d_model = 16
            args.d_ff = 32
            args.dropout = 0.3
            args.fc_dropout = 0.3
            args.head_dropout = 0
            args.patch_len = 16
            args.stride = 8
        elif model_name == 'CATS':
            args.d_model = 128
            args.d_ff = 256
            args.n_heads = 16
            args.QAM_end = 0.2
            args.patch_len = 24
            args.stride = 24
            args.d_layers = 3

    elif data_name == 'ETTm2':
        args.data_path = 'ETTm2.csv'
        args.data = 'ETTm2'
        args.model_id = 'ETTm2'
        args.enc_in = 7
        args.n_heads = 16
        args.d_model = 128
        args.d_ff = 256
        args.dropout = 0.2
        args.fc_dropout = 0.2
        args.head_dropout = 0
        args.batch_size = 128
        args.learning_rate = 0.0001
        args.e_layers = 3
        if model_name == 'TimeMixer':
            args.down_sampling_layers = 3
            args.down_sampling_window = 2
            args.d_model = 32
            args.d_ff = 32
        elif model_name == 'PatchTST':
            args.d_model = 128
            args.d_ff = 256
        elif model_name == 'PDF':
            args.n_heads = 16
            args.d_model = 48
            args.d_ff = 48
            args.period = [102, 720]
            args.patch_len = [16, 24]
            args.stride = [16, 24]
            args.kernel_list = [3, 7, 11]
        elif model_name == 'CARD':
            args.e_layers = 2
            args.n_heads = 2
            args.d_model = 16
            args.d_ff = 32
            args.dropout = 0.3
            args.fc_dropout = 0.3
            args.head_dropout = 0
            args.patch_len = 16
            args.stride = 8
        elif model_name == 'CATS':
            args.d_model = 256
            args.d_ff = 512
            args.n_heads = 16
            args.QAM_end = 0.2
            args.patch_len = 24
            args.stride = 24
            args.query_independence = True
            args.d_layers = 3

    elif data_name == 'exchange_rate':
        args.data_path = 'exchange_rate.csv'
        args.data = 'custom'
        args.model_id = 'exchange_rate'
        args.enc_in = 8
        args.n_heads = 4
        args.d_model = 128
        args.d_ff = 128
        args.dropout = 0.3
        args.fc_dropout = 0.3
        args.head_dropout = 0
        args.batch_size = 32
        args.learning_rate = 0.0025
        args.e_layers = 3
        if model_name == 'TimeMixer':
            args.down_sampling_layers = 3
            args.down_sampling_window = 2
            args.d_model = 16
            args.d_ff = 32
            args.top_k = 5
        elif model_name == 'PatchTST':
            args.d_model = 16
            args.d_ff = 128
        elif model_name == 'PDF':
            args.n_heads = 16
            args.d_model = 48
            args.d_ff = 48
            args.period = [48, 90, 102, 360, 720]
            args.patch_len = [3, 6, 16, 32, 48]
            args.stride = [3, 6, 16, 32, 48]
            args.kernel_list = [5, 7, 11, 15]
        elif model_name == 'CARD':
            args.e_layers = 2
            args.n_heads = 2
            args.d_model = 16
            args.d_ff = 32
            args.dropout = 0.3
            args.fc_dropout = 0.3
            args.head_dropout = 0
            args.patch_len = 16
            args.stride = 8
        elif model_name == 'CATS':
            args.d_model = 256
            args.d_ff = 512
            args.n_heads = 16
            args.QAM_end = 0.2
            args.patch_len = 24
            args.stride = 24
            args.query_independence = True
            args.d_layers = 3

    elif data_name == 'weather':
        args.data_path = 'weather.csv'
        args.model_id = 'weather'
        args.data = 'custom'
        args.enc_in = 21
        args.n_heads = 16
        args.d_model = 128
        args.d_ff = 256
        args.dropout = 0.2
        args.fc_dropout = 0.2
        args.head_dropout = 0
        args.batch_size = 64
        args.learning_rate = 0.0001
        args.e_layers = 3
        if model_name == 'TimeMixer':
            args.down_sampling_layers = 3
            args.down_sampling_window = 2
            args.d_model = 16
            args.d_ff = 32
        elif model_name == 'PatchTST':
            args.d_model = 128
            args.d_ff = 256
        elif model_name == 'PDF':
            args.n_heads = 16
            args.d_model = 64
            args.d_ff = 128
            args.period = [144, 180, 720]
            args.patch_len = [16, 16, 24]
            args.stride = [16, 16, 24]
            args.kernel_list = [3, 7, 11]
        elif model_name == 'CARD':
            args.e_layers = 2
            args.n_heads = 2
            args.d_model = 16
            args.d_ff = 32
            args.dropout = 0.3
            args.fc_dropout = 0.3
            args.head_dropout = 0
            args.patch_len = 16
            args.stride = 8
        elif model_name == 'CATS':
            args.d_model = 128
            args.d_ff = 256
            args.n_heads = 16
            args.QAM_end = 0.2
            args.patch_len = 24
            args.stride = 24
            args.d_layers = 3
            args.query_independence = True

    elif data_name == 'traffic':
        args.data_path = 'traffic.csv'
        args.data = 'custom'
        args.model_id = 'traffic'
        args.enc_in = 862
        args.n_heads = 16
        args.d_model = 128
        args.d_ff = 256
        args.dropout = 0.2
        args.fc_dropout = 0.2
        args.head_dropout = 0
        args.batch_size = 32
        args.learning_rate = 0.0001
        args.e_layers = 3
        if model_name == 'TimeMixer':
            args.down_sampling_layers = 3
            args.down_sampling_window = 2
            args.d_model = 32
            args.d_ff = 64
        elif model_name == 'PatchTST':
            args.d_model = 128
            args.d_ff = 256
        elif model_name == 'PDF':
            args.n_heads = 32
            args.d_model = 128
            args.d_ff = 256
            args.period = [24]
            args.patch_len = [1]
            args.stride = [1]
            args.kernel_list = [3, 7, 9, 11]
        elif model_name == 'CARD':
            args.e_layers = 2
            args.n_heads = 16
            args.d_model = 128
            args.d_ff = 256
            args.dropout = 0.2
            args.fc_dropout = 0.2
            args.head_dropout = 0
            args.patch_len = 16
            args.stride = 8
        elif model_name == 'CATS':
            args.d_model = 128
            args.d_ff = 256
            args.n_heads = 16
            args.QAM_end = 0.2
            args.patch_len = 24
            args.stride = 24
            args.d_layers = 3
            args.query_independence = True

    elif data_name == 'electricity':
        args.data_path = 'electricity.csv'
        args.data = 'custom'
        args.model_id = 'electricity'
        args.enc_in = 321
        args.n_heads = 16
        args.d_model = 128
        args.d_ff = 256
        args.dropout = 0.2
        args.fc_dropout = 0.2
        args.head_dropout = 0
        args.batch_size = 32
        args.learning_rate = 0.0001
        args.e_layers = 3
        if model_name == 'TimeMixer':
            args.down_sampling_layers = 3
            args.down_sampling_window = 2
            args.d_model = 16
            args.d_ff = 32
        elif model_name == 'PatchTST':
            args.d_model = 128 // 2
            args.d_ff = 256 // 2
            args.n_heads = 16 // 2
        elif model_name == 'PDF':
            args.n_heads = 32
            args.d_model = 128
            args.d_ff = 256
            args.period = [8, 12, 24, 180, 720]
            args.patch_len = [1, 2, 3, 16, 48]
            args.stride = [1, 2, 3, 16, 48]
            args.kernel_list = [3, 5, 7, 7]
        elif model_name == 'CARD':
            args.e_layers = 3
            args.n_heads = 8
            args.d_model = 128
            args.d_ff = 256
            args.dropout = 0.2
            args.fc_dropout = 0.2
            args.head_dropout = 0
            args.patch_len = 16
            args.stride = 8
        elif model_name == 'CATS':
            args.d_model = 128
            args.d_ff = 256
            args.n_heads = 16
            args.QAM_end = 0.2
            args.patch_len = 24
            args.stride = 24
            args.d_layers = 3
            args.query_independence = True
        elif model_name == 'TimeXer':
            # electricity has per-dataset TimeXer override (but global overrides below)
            args.d_model = 128
            args.d_ff = 256
            args.n_heads = 16

    # ======================== Global model overrides (lines 2414-2649) ========================
    # These run AFTER per-dataset blocks in original script
    if model_name == 'ModernTCN':
        args.stem_ratio = 6
        args.use_multi_scale = False
        args.dims = [64, 64, 64, 64]
    if model_name == 'TimeXer':
        args.d_model = 128
        args.d_ff = 128
        args.e_layers = 2
        args.d_layers = 1
        args.expand = 2
        args.d_conv = 4
        args.top_k = 5
        args.num_kernels = 6
        args.n_heads = 8
        args.moving_avg = 25
        args.factor = 1
        args.channel_independence = 1
        args.dropout = 0.1

    # ======================== Common overrides (lines 2550-2582) ========================
    args.enc_in = 1
    args.enc_out = 1
    args.c_out = 1
    args.c_in = 1
    args.dec_in = 1
    args.query_independence = True

    # ======================== LSINet specific (lines 2586-2616) ========================
    if model_name == 'LSINet':
        args.d_v = 128
        args.d_ff = 256
        args.n_heads = 16
        args.e_layers = 3
        args.d_layers = 1
        args.kernel_size = 25
        args.dropout = 0.2
        args.fc_dropout = 0.2
        args.head_dropout = 0
        args.maximum_patch_num = 32
        args.stride = int(args.seq_len / args.maximum_patch_num)
        args.patch_len = int(args.seq_len / args.maximum_patch_num) * 2
        args.reduce_dim = 64
        args.n_msim = 1
        args.n_heads_msim = 4
        args.n_msim_residual = 1
        args.sparse_rate = 0.15
        args.Self_Attention_Mechanism = False
        args.MSIM = True
        args.n_heads_sam = args.n_heads
        args.resdual_block = True
        args.var_individual = False
        args.var_decomp = False
        args.SpIntervel = 3

    # ======================== SEMixer specific (lines 2617-2649) ========================
    elif model_name == 'SEMixer':
        args.d_model = 128
        args.d_ff = 256
        args.n_heads = 16
        args.e_layers = 3
        args.d_layers = 1
        args.kernel_size = 25
        args.dropout = 0.2
        args.fc_dropout = 0.2
        args.head_dropout = 0
        args.reduce_dim = 64
        args.maximum_patch_num = 16
        args.stride = int(args.seq_len / args.maximum_patch_num)
        args.patch_len = int(args.seq_len / args.maximum_patch_num)
        args.multi_scale = True
        args.eib_num = 1
        args.eib_num_1scale = 1
        args.scale_factors = [1, 2, 4, 8]
        args.sample_num = 5
        args.Random_Attention_Mechanism = True
        args.self_attn = False
        args.prob_attn = False
        args.logsparse = False
        args.peformer_attn = False
        args.reformer_attn = False
        args.autocorrelation = False
        args.fed_fourier_attn = False
        args.var_individual = False
        args.var_decomp = False
        args.Self_Attention_Mechanism = False
        args.connection_probability = 0.85

    # ======================== MLF specific ========================
    elif model_name == 'MLF':
        args.fixed_patch_num = 6
        args.MAP_alpha = 2
        args.MAP = True
        args.embed_dim = 3
        args.redundancy_scaling = True
        args.scal_all = [args.seq_len // 2, args.seq_len]
        args.patchLen_stride_all = []
        args.equal_patch_len = []
        for i, period_s in enumerate(args.scal_all):
            args.patchLen_stride_all.append(
                [int(period_s / args.fixed_patch_num) * args.MAP_alpha,
                 int(period_s / args.fixed_patch_num)])
            args.equal_patch_len.append(
                int(period_s / args.fixed_patch_num) * args.MAP_alpha)
        args.max_patch_len = args.patchLen_stride_all[-1][0]
        args.patch_squeeze = False
        args.squeeze_factor = [4 for _ in range(len(args.scal_all))]
        args.context_window = None
        args.threshold_patch_num = 7
        args.D_norm = False
        args.revin_norm = True
        args.patch_pad = True
        args.activation_tag = True
        args.LWI = True


def infer_patch_len_from_checkpoint(ckpt_path, model_name):
    """Infer patch_len from checkpoint state_dict shapes for PatchTST/TimeXer"""
    sd = torch.load(ckpt_path, map_location='cpu')
    if model_name == 'PatchTST':
        for key in sd:
            if 'W_P.weight' in key:
                return sd[key].shape[1]
    elif model_name == 'TimeXer':
        for key in sd:
            if 'value_embedding.weight' in key and 'en_embedding' in key:
                return sd[key].shape[1]
    return None


def run_inference(model, model_instance, batch_x, batch_y, args, batch_size=256):
    """Run batched inference, return predictions array"""
    N = batch_x.shape[0]
    pred_chunks = []
    device = model_instance.device

    with torch.no_grad():
        for b_start in range(0, N, batch_size):
            b_end = min(b_start + batch_size, N)
            bx = batch_x[b_start:b_end].float().to(device)
            by = batch_y[b_start:b_end].float().to(device)
            bx_mark = torch.zeros_like(bx).to(device).repeat(1, 1, 4)
            by_mark = torch.zeros_like(by).to(device).repeat(1, 1, 4)
            d_inp = torch.zeros_like(by[:, -args.pred_len:, :]).float()
            d_inp = torch.cat([by[:, :args.label_len, :], d_inp], dim=1).float()

            try:
                out = model(bx, bx_mark, d_inp, by_mark)
            except:
                out = model(bx)

            if args.model == 'LSINet':
                out, _ = out
            elif args.model == 'MLF':
                out, _, _ = out

            pred_chunks.append(out[:, -args.pred_len:, :].detach().cpu().numpy())

    return np.concatenate(pred_chunks, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models_root', type=str, required=True,
                        help='External forecasting-model source directory containing models/')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--dataset', type=str, required=True, choices=ALL_DATASETS)
    parser.add_argument('--models', nargs='+', default=ALL_MODELS)
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'])
    parser.add_argument('--base_path', type=str, default=BASE_PATH,
                        help='Base path for data (MTS or UTS)')
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='Base path for checkpoints (defaults to base_path)')
    cli_args = parser.parse_args()

    cli_args.models_root = os.path.abspath(os.path.expanduser(cli_args.models_root))
    if not os.path.isdir(os.path.join(cli_args.models_root, 'models')):
        parser.error('--models_root must point to an existing directory containing models/')
    sys.path.insert(0, cli_args.models_root)

    dataset_name = cli_args.dataset
    base_args = setup_base_args(gpu=cli_args.gpu)
    data_path = os.path.join(cli_args.base_path, dataset_name)

    print(f"=== Dataset: {dataset_name} ===")

    for model_name in cli_args.models:
        if model_name not in ALL_MODELS:
            print(f"Skipping unknown model: {model_name}")
            continue

        model_dir = os.path.join(data_path, model_name, SEED_NAME)
        ckpt_base = cli_args.ckpt_path if cli_args.ckpt_path else cli_args.base_path
        ckpt_dir = os.path.join(ckpt_base, dataset_name, model_name, SEED_NAME)
        ckpt_path = os.path.join(ckpt_dir, f'checkpoint_{SEQ_LEN}_{PRED_LEN}.pth')
        if not os.path.exists(ckpt_path):
            # Fallback: try MTS path for checkpoint
            ckpt_path_mts = os.path.join(BASE_PATH, dataset_name, model_name, SEED_NAME, f'checkpoint_{SEQ_LEN}_{PRED_LEN}.pth')
            if os.path.exists(ckpt_path_mts):
                ckpt_path = ckpt_path_mts
            else:
                print(f"[SKIP] {model_name}: checkpoint not found at {ckpt_path}")
                continue

        for split in cli_args.splits:
            save_name = f'{split}_pred_x_all_{SEQ_LEN}_{PRED_LEN}.npy'
            save_path = os.path.join(model_dir, save_name)
            if os.path.exists(save_path):
                existing = np.load(save_path)
                bx_check = os.path.join(model_dir, f'{split}_batch_x_all_{SEQ_LEN}_{PRED_LEN}.npy')
                if os.path.exists(bx_check):
                    expected_n = np.load(bx_check, mmap_mode='r').shape[0]
                    if existing.shape[0] == expected_n:
                        print(f"[SKIP] {model_name}/{split}: already exists with correct shape {existing.shape}")
                        continue
                    else:
                        print(f"[REDO] {model_name}/{split}: shape {existing.shape} != expected ({expected_n},...)")
                else:
                    print(f"[SKIP] {model_name}/{split}: already exists with shape {existing.shape}")
                    continue

            # Load data
            bx_path = os.path.join(model_dir, f'{split}_batch_x_all_{SEQ_LEN}_{PRED_LEN}.npy')
            by_path = os.path.join(model_dir, f'{split}_batch_y_all_{SEQ_LEN}_{PRED_LEN}.npy')
            if not os.path.exists(bx_path) or not os.path.exists(by_path):
                print(f"[SKIP] {model_name}/{split}: batch_x/y not found")
                continue

            batch_x = torch.from_numpy(np.load(bx_path))
            batch_y = torch.from_numpy(np.load(by_path))
            print(f"[RUN] {model_name}/{split}: batch_x={batch_x.shape}, batch_y={batch_y.shape}")

            # Build model & load weights
            args = argparse.Namespace(**vars(base_args))
            set_dataset_model_args(args, dataset_name, model_name)

            # Infer patch_len from checkpoint for PatchTST/TimeXer to avoid shape mismatch
            if model_name in ('PatchTST', 'TimeXer'):
                inferred_pl = infer_patch_len_from_checkpoint(ckpt_path, model_name)
                if inferred_pl is not None:
                    args.patch_len = inferred_pl
                    args.stride = inferred_pl
                    print(f"  Inferred patch_len={inferred_pl} from checkpoint")

            model_instance = Model_Basic(args)
            model = model_instance.bulid_model()
            model.load_state_dict(torch.load(ckpt_path, map_location=f'cuda:{args.gpu}'))
            model.eval()

            # Run inference
            pred = run_inference(model, model_instance, batch_x, batch_y, args)
            np.save(save_path, pred)
            print(f"[DONE] {model_name}/{split}: saved {pred.shape} -> {save_path}")

            # Free GPU memory
            del model, model_instance
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
