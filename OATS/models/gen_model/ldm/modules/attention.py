# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat

from ldm.modules.diffusionmodules.util import checkpoint


def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0., use_pam=False):
        super().__init__()
        self.use_pam = use_pam
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        if len(context.shape) == 2:
            k = self.to_k(context)[:,None]
            v = self.to_v(context)[:,None]
        else:
            k = self.to_k(context)
            v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        # print("q, k, v shapes:", q.shape, k.shape, v.shape)

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            if self.use_pam:
                mask_of_mask = torch.where(mask > 0, torch.zeros_like(mask), torch.ones_like(mask))
                max_neg_value = -torch.finfo(mask.dtype).max
                mask = mask_of_mask * max_neg_value + mask
                sim = sim + mask
            # sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=False, use_pam=False):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention
        self.ff = FeedForward(dim+64, dim_out=dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout, use_pam=use_pam)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None, mask=None, emb_class=None):
        return checkpoint(self._forward, (x, context, mask, emb_class), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None, mask=None, emb_class=None):
        # if emb_class is not None:
        #     print("I received emb_class:", emb_class.shape)
        # else:
        #     print("emb_class is None, no class condition")
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context, mask=mask) + x

        if emb_class is not None:
            emb_class = emb_class[:, None, :]   # → (B, 1, D), broadcasts across N
            emb_class_expanded = emb_class.expand(-1, x.shape[1], -1)
            # print(x.shape, emb_class_expanded.shape)
            x_mod = torch.cat((self.norm3(x), emb_class_expanded), dim=-1)        # → (B, N, D)
        else:
            x_mod = torch.cat((self.norm3(x), torch.zeros(x.shape[0], x.shape[1], 64).to("cuda")), dim=-1)
        x = self.ff(x_mod) + x
        # print("x.shape after ff:", x.shape)
        return x

    
class Spatial1DTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None, use_pam=False):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        self.proj_in = nn.Conv1d(in_channels,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim, use_pam=use_pam)
                for d in range(depth)]
        )

        self.proj_out = zero_module(nn.Conv1d(inner_dim,
                                              in_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))

    def forward(self, x, context=None, mask=None, emb_class=None):
        # note: if no context is given, cross-attention defaults to self-attention
        # if emb_class is not None:
        #     print("I received emb_class:", emb_class.shape)
        # else:
        #     print("emb_class is None, no class condition")
        b, c, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c w -> b w c')
        for block in self.transformer_blocks:
            x = block(x, context=context, mask=mask, emb_class=emb_class)
        x = rearrange(x, 'b w c -> b c w', w=w)
        x = self.proj_out(x)
        return x + x_in
