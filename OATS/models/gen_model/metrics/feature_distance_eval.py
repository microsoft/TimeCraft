# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
import torch
from torch import nn
from scipy.stats import entropy
from sklearn.metrics.pairwise import rbf_kernel

def cal_distances(gt, sp):
    gt = gt[~np.isnan(gt)]
    # gt = gt[gt<np.nanpercentile(gt, 99.5)]
    sp = sp[~np.isnan(sp)]
    hist_real, edge_real = np.histogram(gt, density=True, bins=50)
    hist_gen, _ = np.histogram(sp, density=True, bins=edge_real)
    kl = entropy(hist_real, hist_gen+1e-9)
    return {
        'kl': kl
        }

def calculate_mmd(x, y, kernel=rbf_kernel):
    """
    Calculate the Maximum Mean Discrepancy (MMD) between two samples.
    
    Args:
    x, y: torch.Tensor, shape (n_samples, n_features)
    kernel: function, the kernel function to use (default: Gaussian kernel)
    
    Returns:
    mmd: float, the MMD between x and y
    """
    xx = kernel(x, x)
    yy = kernel(y, y)
    xy = kernel(x, y)
    
    mmd = xx.mean() + yy.mean() - 2 * xy.mean()
    return mmd.item()

def mmd_metric(real_data, generated_data):
    """
    Compute MMD metric between real and generated data using both Gaussian and RBF kernels.
    
    Args:
    real_data: numpy array of shape (n_samples, seq_len, n_features)
    generated_data: numpy array of shape (n_samples, seq_len, n_features)
    
    Returns:
    dict: containing MMD values for both Gaussian and RBF kernels
    """
    # Flatten the time dimension
    real_flat = real_data.reshape(real_data.shape[0], -1)
    gen_flat = generated_data.reshape(generated_data.shape[0], -1)
    
    # Convert to PyTorch tensors for Gaussian kernel
    real_tensor = torch.FloatTensor(real_flat)
    gen_tensor = torch.FloatTensor(gen_flat)
    
    # Calculate MMD with Gaussian kernel and rbf kernel
    mmd_rbf = calculate_mmd(real_tensor, gen_tensor, kernel=rbf_kernel)
    
    return {
        'mmd_rbf': mmd_rbf
        }
    
def get_flat_distance(ori_data, gen_data):
    result_dict = {}
    flat_ori = ori_data.flatten()
    flat_gen = gen_data.flatten()
    flat_result = cal_distances(flat_ori, flat_gen)
    for key, value in flat_result.items():
        result_dict[f'flat_{key}'] = value
    return result_dict

def histogram_torch(x, n_bins, density=True):
    a, b = x.min().item(), x.max().item()
    b = b+1e-2 if b == a else b
    bins = torch.linspace(a, b, n_bins+1)
    delta = bins[1]-bins[0]
    count = torch.histc(x, bins=n_bins, min=a, max=b).float()
    if density:
        count = count / delta / float(x.shape[0] * x.shape[1])
    return count, bins

class Loss(nn.Module):
    def __init__(self, name, reg=1.0, transform=lambda x: x, threshold=10., backward=False, norm_foo=lambda x: x):
        super(Loss, self).__init__()
        self.name = name
        self.reg = reg
        self.transform = transform
        self.threshold = threshold
        self.backward = backward
        self.norm_foo = norm_foo

    def forward(self, x_fake):
        self.loss_componentwise = self.compute(x_fake)
        return self.reg * self.loss_componentwise.mean()

    def compute(self, x_fake):
        raise NotImplementedError()

    @property
    def success(self):
        return torch.all(self.loss_componentwise <= self.threshold)

class HistoLoss(Loss):

    def __init__(self, x_real, n_bins, **kwargs):
        super(HistoLoss, self).__init__(**kwargs)
        self.densities = list()
        self.locs = list()
        self.deltas = list()
        for i in range(x_real.shape[2]):
            tmp_densities = list()
            tmp_locs = list()
            tmp_deltas = list()
            # Exclude the initial point
            for t in range(x_real.shape[1]):
                x_ti = x_real[:, t, i].reshape(-1, 1)
                d, b = histogram_torch(x_ti, n_bins, density=True)
                tmp_densities.append(nn.Parameter(d).to(x_real.device))
                delta = b[1:2] - b[:1]
                loc = 0.5 * (b[1:] + b[:-1])
                tmp_locs.append(loc)
                tmp_deltas.append(delta)
            self.densities.append(tmp_densities)
            self.locs.append(tmp_locs)
            self.deltas.append(tmp_deltas)

    def compute(self, x_fake):
        loss = list()

        def relu(x):
            return x * (x >= 0.).float()

        for i in range(x_fake.shape[2]):
            for t in range(x_fake.shape[1]):
                loc = self.locs[i][t].view(1, -1).to(x_fake.device)
                x_ti = x_fake[:, t, i].contiguous(
                ).view(-1, 1).repeat(1, loc.shape[1])
                dist = torch.abs(x_ti - loc)
                left_counter = ((self.deltas[i][t].to(x_fake.device) / 2. - (loc - x_ti)) == 0.).float()
                counter = (relu(self.deltas[i][t].to(x_fake.device) / 2. - dist) > 0.).float() + left_counter
                density = counter.mean(0) / self.deltas[i][t].to(x_fake.device)
                abs_metric = torch.abs(
                    density - self.densities[i][t].to(x_fake.device))
                loss.append(torch.mean(abs_metric, 0))
        loss_componentwise = torch.stack(loss)
        return loss_componentwise


def get_mdd_eval(ori_data, generated_data, n_bins=20):
    x_real = torch.Tensor(ori_data)
    x_fake = torch.Tensor(generated_data)
    mdd = (HistoLoss(x_real, n_bins=n_bins, name='marginal_distribution')(x_fake)).detach().cpu().numpy()
    
    return mdd
