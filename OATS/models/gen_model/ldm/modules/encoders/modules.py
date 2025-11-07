# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn as nn
from einops import repeat
import copy
       
# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

class ResBlockTime(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, bn=False):
        super(ResBlockTime, self).__init__()

        if mid_channels is None:
            mid_channels = out_channels

        layers = [
            nn.ReLU(),
            nn.Conv1d(in_channels, mid_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(mid_channels, out_channels,
                      kernel_size=1, stride=1, padding=0)
        ]
        if bn:
            layers.insert(2, nn.BatchNorm1d(out_channels))
        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.convs(x)

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)

class DomainUnifiedEncoder(nn.Module):
    '''
    The input are encoded into two parts, invariant part and specific part. The specific part is generated attending to a random initialized latent vector pool.
    The length of the two part are equal in this implementation.
    '''
    def __init__(self, dim, window, num_channels=3, latent_dim=32, bn=True, **kwargs):
        super().__init__()
        dim_out = latent_dim
        flatten_dim = int(dim * window / 4)
        self.in_encoder = nn.Sequential(
            nn.Conv1d(num_channels, dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(dim, dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True)
            )

        self.out_encoder = nn.Sequential(
            ResBlockTime(dim, dim, bn=bn),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            ResBlockTime(dim, dim, bn=bn),
            View((-1, flatten_dim)),                  # batch_size x 2048
            nn.Linear(flatten_dim, dim_out)
        )
            
    def forward(self, x):
        h = self.in_encoder(x)
        mask = None

        out = self.out_encoder(h)[:,None]   # b, 1, d
        return out, mask

class DomainUnifiedPrototyper(nn.Module):
    '''
    The input are encoded into two parts, invariant part and specific part. The specific part is generated attending to a random initialized latent vector pool.
    The length of the two part are equal in this implementation.
    '''
    def __init__(self, dim, window, num_latents=16, num_channels=3, latent_dim=32, bn=True, **kwargs):
        super().__init__()
        self.num_latents = num_latents
        self.latent_dim = latent_dim
        flatten_dim = int(dim * window / 4)
        self.share_encoder = nn.Sequential(
            nn.Conv1d(num_channels, dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(dim, dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True)
            )
        self.latents = nn.Parameter(torch.empty(num_latents, self.latent_dim), requires_grad=False)
        nn.init.orthogonal_(self.latents)
        self.init_latents = copy.deepcopy(self.latents.detach())
        self.mask_ffn = nn.Sequential(
            ResBlockTime(dim, dim, bn=bn),
            View((-1, flatten_dim)),                  # batch_size x 2048
            nn.Linear(flatten_dim, self.num_latents),
        )
        self.sigmoid = nn.Sigmoid()
            
    def forward(self, x):
        b = x.shape[0]
        h = self.share_encoder(x)
        mask = None

        latents = repeat(self.latents, 'n d -> b n d', b = b)
        mask_logit = self.mask_ffn(h)
        mask = mask_logit  # soft assign
                    
        out = latents  #  mask
        return out, mask
        
