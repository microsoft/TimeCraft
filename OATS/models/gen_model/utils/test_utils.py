# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
import torch
from pathlib import Path
from utils.pkl_utils import save_pkl
from metrics.metrics_sets import run_metrics, calculate_one
from ldm.data.tsg_dataset import TSGDataset
import os

data_root = os.environ['DATA_ROOT']

def test_model_with_dp(model, data, trainer, opt, logdir):
    if trainer.callbacks[-1].best_model_path:
        best_ckpt_path = trainer.callbacks[-1].best_model_path
        print(f"Loading best model from {best_ckpt_path} for sampling")
        model.init_from_ckpt(best_ckpt_path)
    model = model.cuda()
    model.eval()
    save_path = Path(logdir) / 'generated_samples'
    save_path.mkdir(exist_ok=True, parents=True)
    seq_len = data.window
    num_dp = 100  # number of samples for constructingdomain prompts
    all_metrics = {}
    for dataset in data.norm_train_dict:
        dataset_data = TSGDataset({dataset: data.norm_train_dict[dataset]})
        dataset_samples = []
        for idx in np.random.randint(dataset_data.__len__(),size=num_dp):  # randomly sample num_dp samples from the dataset
            dataset_samples.append(dataset_data.__getitem__(idx)['context'])
        dataset_samples = np.vstack(dataset_samples)
            
        x = torch.tensor(dataset_samples).to('cuda').float().unsqueeze(1)[:num_dp]
        c, mask = model.get_learned_conditioning(x, return_mask=True)
        repeats = int(1000 / num_dp) if not opt.debug else 1

        if c is None:
            mask_repeat = None
            cond = None
        elif mask is None:
            cond = torch.repeat_interleave(c, repeats, dim=0)
            mask_repeat = None
        else:
            cond = torch.repeat_interleave(c, repeats, dim=0)
            mask_repeat = torch.repeat_interleave(mask, repeats, dim=0)

        all_gen = []
        for _ in range(5 if not opt.debug else 1):  # iterate to reduce maximum memory usage
            samples, _ = model.sample_log(cond=cond, batch_size=1000 if not opt.debug else 100, ddim=False, cfg_scale=1, mask=mask_repeat)
            norm_samples = model.decode_first_stage(samples).detach().cpu().numpy()
            inv_samples = data.inverse_transform(norm_samples, data_name=dataset)
            all_gen.append(inv_samples)
        generated_data = np.vstack(all_gen).transpose(0, 2, 1)
        # save data in original scale, for fairness in evaluation
        tmp_name = f'{dataset}_{seq_len}_generation'
        np.save(save_path / f'{tmp_name}.npy', generated_data)
        all_metrics = run_metrics(data_name=dataset, seq_len=seq_len, model_name=tmp_name, gen_data=generated_data, scale='zscore', exist_dict=all_metrics)
    print(all_metrics)
    save_pkl(all_metrics, Path(logdir) / 'metric_dict.pkl')
    

def test_model_uncond(model, data, trainer, opt, logdir):
    if trainer.callbacks[-1].best_model_path:
        best_ckpt_path = trainer.callbacks[-1].best_model_path
        print(f"Loading best model from {best_ckpt_path} for sampling")
        model.init_from_ckpt(best_ckpt_path)
    model = model.cuda()
    model.eval()
    save_path = Path(logdir) / 'generated_samples'
    save_path.mkdir(exist_ok=True, parents=True)
    seq_len = data.window
    all_metrics = {}
    for dataset in data.norm_train_dict:            

        all_gen = []
        for _ in range(5 if not opt.debug else 1):
            samples, _ = model.sample_log(cond=None, batch_size=1000 if not opt.debug else 100, ddim=False, cfg_scale=1)
            norm_samples = model.decode_first_stage(samples).detach().cpu().numpy()
            inv_samples = data.inverse_transform(norm_samples, data_name=dataset)
            all_gen.append(inv_samples)
        generated_data = np.vstack(all_gen).transpose(0, 2, 1)
        # save data in original scale. for fair use in evaluation
        tmp_name = f'{dataset}_{seq_len}_uncond_generation'
        np.save(save_path / f'{tmp_name}.npy', generated_data)
        all_metrics = run_metrics(data_name=dataset, seq_len=seq_len, model_name=tmp_name, gen_data=generated_data, scale='zscore', exist_dict=all_metrics)
    print(all_metrics)
    save_pkl(all_metrics, Path(logdir) / 'metric_dict.pkl')
    
def zero_shot_k_repeat(samples, model, train_data_module, num_gen_samples=1000):
    data = train_data_module
    k_samples = samples.transpose(0,2,1)
    k = k_samples.shape[0]
    normalizer = data.fit_normalizer(k_samples)

    norm_k_samples = data.transform(k_samples, normalizer=normalizer)

    x = torch.tensor(norm_k_samples).float().to('cuda')
    c, mask = model.get_learned_conditioning(x, return_mask=True)

    repeats = int(num_gen_samples / k)
    extra = num_gen_samples - repeats * k
    
    cond = torch.repeat_interleave(c, repeats, dim=0)
    cond = torch.cat([cond, c[:extra]], dim=0)
    mask_repeat = torch.repeat_interleave(mask, repeats, dim=0)
    mask_repeat = torch.cat([mask_repeat, mask[:extra]], dim=0)
    
    samples, z_denoise_row = model.sample_log(cond=cond, batch_size=cond.shape[0], ddim=False, cfg_scale=1, mask=mask_repeat)
    norm_samples = model.decode_first_stage(samples).detach().cpu().numpy()
    inv_samples = data.inverse_transform(norm_samples, normalizer=normalizer)
    gen_data = inv_samples.transpose(0,2,1)
    
    return gen_data, k_samples.transpose(0,2,1)

def merge_dicts(dicts):
    result = {}
    for d in dicts:
        for k, v in d.items():
            result[k] = v
    return result

def test_model_unseen(model, data, trainer, opt, logdir):
    all_metrics = {}
    seq_len = opt.seq_len
    for data_name in ['stock', 'web']:
        data_result_dicts = []
        uni_ori_data = np.load(f'{data_root}/ts_data/new_zero_shot_data/{data_name}_{seq_len}_test_sample.npy')
        if data_name == 'web':
            uni_ori_data = uni_ori_data[uni_ori_data<np.percentile(uni_ori_data,99)]
        uni_data_mean, uni_data_std = np.mean(uni_ori_data), np.std(uni_ori_data)
        uni_data_sub, uni_data_div = uni_data_mean, uni_data_std + 1e-7
        uni_scaled_ori = (uni_ori_data - uni_data_sub) / uni_data_div
        print(data_name, 'univar', uni_scaled_ori.shape)

        scaled_ori = uni_scaled_ori
        
        total_samples = 2000
        for k in [3, 10, 100]: 
            k_samples = np.load(f'{data_root}/ts_data/new_zero_shot_data/{data_name}_{seq_len}_k_{k}_sample.npy')
            for i in range(1):
                gen_data, _ = zero_shot_k_repeat(k_samples, model=model, train_data_module=data, num_gen_samples=total_samples)
                np.save(logdir/f"generated_samples/{data_name}_{seq_len}_k{k}_repeat_gen.npy", gen_data)
                this_metrics = calculate_one(gen_data.squeeze(), scaled_ori.squeeze(), '', i, f"{data_name}_{k}", seq_len, uni_data_sub, uni_data_div, total_samples)
                data_result_dicts.append(this_metrics)
                

        data_metrics = merge_dicts(data_result_dicts)
        all_metrics.update(data_metrics)
    print(all_metrics)
    save_pkl(all_metrics, Path(logdir) / 'unseen_domain_metric_dict.pkl')
