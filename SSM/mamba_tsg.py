"""
Mamba-based Time Series Generator

Based on "Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (Gu & Dao 2023)
https://arxiv.org/abs/2312.00752

Key innovations:
1. Selective scan: B, C, and Δ (delta/step size) are input-dependent
2. Hardware-efficient parallel scan
3. Better at capturing discrete patterns and long-range dependencies

Discretization (Zero-Order Hold):
    A_bar = exp(Δ * A)
    B_bar = (exp(Δ * A) - I) * A^{-1} * B  ≈  Δ * B  (simplified)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SelectiveSSM(nn.Module):
    """
    Selective State Space Model (core of Mamba)

    Key difference from S4: B, C, and Δ are computed from the input,
    making the state transitions content-aware (selective).

    Parameters:
        d_model: Input/output dimension
        d_state: SSM state dimension (N in paper, typically 16)
        d_conv: Local convolution width (typically 4)
        expand: Expansion factor for inner dimension (typically 2)
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: str = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)

        # dt_rank: dimension for delta projection
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        # Input projection: x -> (z, x) where z is the gate
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # 1D convolution for local context (causal)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,  # depthwise
            bias=True,
        )

        # Selective parameters projection: x -> (Δ, B, C)
        # This is the KEY difference from S4 - these are input-dependent
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + d_state * 2, bias=False
        )

        # Delta (Δ) projection from dt_rank to d_inner
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Initialize dt_proj bias for proper Δ range
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)

        # Initialize dt bias to map softplus output to [dt_min, dt_max]
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        )
        # Inverse of softplus to get bias
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # A parameter: diagonal, initialized to -exp(uniform)
        # Negative ensures stability (eigenvalues in left half-plane)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))  # (d_inner, d_state)

        # D: skip connection (identity-like initialization)
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        """
        Forward pass with selective scan.

        Args:
            x: (batch, seq_len, d_model)
        Returns:
            y: (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape

        # 1. Input projection and split into x and gate z
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x, z = xz.chunk(2, dim=-1)  # Each: (B, L, d_inner)

        # 2. Causal convolution for local context
        x = x.transpose(1, 2)  # (B, d_inner, L)
        x = self.conv1d(x)[:, :, :seq_len]  # Causal: remove future padding
        x = x.transpose(1, 2)  # (B, L, d_inner)
        x = F.silu(x)

        # 3. Compute selective parameters (B, C, Δ) from x
        y = self.selective_scan(x)

        # 4. Gate with z and project output
        y = y * F.silu(z)
        y = self.out_proj(y)

        return y

    def selective_scan(self, x):
        """
        Selective scan with input-dependent B, C, Δ.

        This is the sequential implementation. The parallel associative scan
        version is more efficient but harder to implement correctly.
        """
        batch, seq_len, d_inner = x.shape

        # Get A (negative for stability)
        A = -torch.exp(self.A_log)  # (d_inner, d_state)

        # Project x to get Δ, B, C (selective/input-dependent)
        x_dbl = self.x_proj(x)  # (B, L, dt_rank + 2*d_state)

        # Split into components
        delta, B, C = x_dbl.split([self.dt_rank, self.d_state, self.d_state], dim=-1)

        # Project and apply softplus to get positive Δ
        delta = self.dt_proj(delta)  # (B, L, d_inner)
        delta = F.softplus(delta)  # Ensure positive

        # Selective scan (sequential for clarity)
        # State: (B, d_inner, d_state)
        h = torch.zeros(batch, d_inner, self.d_state, device=x.device, dtype=x.dtype)
        outputs = []

        for t in range(seq_len):
            x_t = x[:, t, :]  # (B, d_inner)
            delta_t = delta[:, t, :]  # (B, d_inner)
            B_t = B[:, t, :]  # (B, d_state)
            C_t = C[:, t, :]  # (B, d_state)

            # Discretization (simplified ZOH):
            # A_bar = exp(Δ * A)  ≈  1 + Δ * A  (first-order approximation)
            # B_bar = Δ * B

            # For each channel in d_inner:
            # h = A_bar * h + B_bar * x
            # y = C * h

            # A_bar: (B, d_inner, d_state)
            delta_A = delta_t.unsqueeze(-1) * A  # (B, d_inner, d_state)
            A_bar = torch.exp(delta_A)  # Proper discretization

            # B_bar: (B, d_inner, d_state)
            # We broadcast x_t and B_t
            delta_B = delta_t.unsqueeze(-1) * B_t.unsqueeze(1)  # (B, d_inner, d_state)

            # State update: h = A_bar * h + delta_B * x
            h = A_bar * h + delta_B * x_t.unsqueeze(-1)

            # Output: y = h @ C (sum over state dimension)
            y_t = (h * C_t.unsqueeze(1)).sum(dim=-1)  # (B, d_inner)

            # Skip connection
            y_t = y_t + self.D * x_t

            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)  # (B, L, d_inner)
        return y

    def step(self, x_t, state):
        """
        Single step for autoregressive generation.

        Args:
            x_t: (B, d_model) input at current timestep
            state: (h, conv_state) tuple
        Returns:
            y_t: (B, d_model) output
            new_state: updated state
        """
        h, conv_state = state

        # Input projection
        xz = self.in_proj(x_t)  # (B, 2*d_inner)
        x, z = xz.chunk(2, dim=-1)

        # Update conv state and apply convolution
        conv_state = torch.roll(conv_state, -1, dims=1)
        conv_state[:, -1, :] = x
        x = (conv_state * self.conv1d.weight.view(self.d_inner, self.d_conv)).sum(dim=1)
        x = x + self.conv1d.bias
        x = F.silu(x)

        # Selective parameters
        x_dbl = self.x_proj(x)
        delta, B, C = x_dbl.split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta))

        # State update
        A = -torch.exp(self.A_log)
        delta_A = delta.unsqueeze(-1) * A
        A_bar = torch.exp(delta_A)
        delta_B = delta.unsqueeze(-1) * B.unsqueeze(1)
        h = A_bar * h + delta_B * x.unsqueeze(-1)

        # Output
        y = (h * C.unsqueeze(1)).sum(dim=-1) + self.D * x
        y = y * F.silu(z)
        y = self.out_proj(y)

        return y, (h, conv_state)

    def init_state(self, batch_size, device):
        """Initialize state for autoregressive generation."""
        h = torch.zeros(batch_size, self.d_inner, self.d_state, device=device)
        conv_state = torch.zeros(batch_size, self.d_conv, self.d_inner, device=device)
        return (h, conv_state)


class MambaBlock(nn.Module):
    """Mamba block with residual connection and normalization."""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = SelectiveSSM(d_model, d_state, d_conv, expand)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return x + self.dropout(self.mamba(self.norm(x)))

    def step(self, x_t, state):
        """Single step for generation."""
        y_t, new_state = self.mamba.step(self.norm(x_t), state)
        return x_t + self.dropout(y_t), new_state


class MambaVAE(nn.Module):
    """
    Mamba-based VAE for time series generation.

    Architecture:
    - Mamba encoder: sequence -> latent distribution
    - VAE sampling: z ~ N(mu, sigma)
    - Mamba decoder: latent -> sequence

    Why this should work better for time series patterns:
    - Selective mechanism can learn to focus on trend/periodicity
    - HiPPO-like state can capture long-range dependencies
    - Input-dependent Δ can adapt to different time scales
    """

    def __init__(
        self,
        seq_len: int = 168,
        d_input: int = 1,
        d_model: int = 64,
        d_state: int = 16,
        d_latent: int = 32,
        n_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.d_input = d_input
        self.d_model = d_model
        self.d_latent = d_latent

        # Input embedding
        self.input_embed = nn.Linear(d_input, d_model)

        # Encoder: Mamba blocks
        self.enc_layers = nn.ModuleList([
            MambaBlock(d_model, d_state, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.enc_norm = nn.LayerNorm(d_model)

        # Latent projections
        self.to_mu = nn.Linear(d_model, d_latent)
        self.to_logvar = nn.Linear(d_model, d_latent)

        # Decoder
        self.latent_to_seq = nn.Linear(d_latent, d_model * seq_len)
        self.dec_layers = nn.ModuleList([
            MambaBlock(d_model, d_state, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.dec_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, d_input)

    def encode(self, x):
        """Encode sequence to latent distribution parameters."""
        # x: (B, L, d_input)
        h = self.input_embed(x)  # (B, L, d_model)

        for layer in self.enc_layers:
            h = layer(h)
        h = self.enc_norm(h)

        # Pool over sequence (use mean)
        h = h.mean(dim=1)  # (B, d_model)

        mu = self.to_mu(h)
        logvar = self.to_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Sample z using reparameterization trick."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z):
        """Decode latent to sequence."""
        batch = z.shape[0]

        # Project latent to sequence
        h = self.latent_to_seq(z)  # (B, d_model * seq_len)
        h = h.view(batch, self.seq_len, self.d_model)

        for layer in self.dec_layers:
            h = layer(h)
        h = self.dec_norm(h)

        return self.output_proj(h)  # (B, L, d_input)

    def forward(self, x):
        """Forward pass for training."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_rec = self.decode(z)
        return x_rec, mu, logvar

    @torch.no_grad()
    def sample(self, n_samples, device='cpu'):
        """Generate new sequences by sampling from prior."""
        self.eval()
        z = torch.randn(n_samples, self.d_latent, device=device)
        return self.decode(z)

    def loss(self, x, x_rec, mu, logvar, kl_weight=0.1):
        """
        VAE loss = Reconstruction + KL divergence.

        Args:
            x: (B, L, d_input) original
            x_rec: (B, L, d_input) reconstructed
            mu, logvar: latent distribution parameters
            kl_weight: weight for KL term (beta-VAE style)
        """
        # Reconstruction loss (MSE)
        rec_loss = F.mse_loss(x_rec, x, reduction='mean')

        # KL divergence against N(0, 1)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        total_loss = rec_loss + kl_weight * kl_loss
        return total_loss, rec_loss, kl_loss


class MambaTimeSeriesGenerator(nn.Module):
    """
    Autoregressive Mamba generator for time series.

    This model generates sequences step-by-step, which can better
    maintain trends and patterns compared to diffusion approaches.
    """

    def __init__(
        self,
        seq_len: int = 168,
        d_input: int = 1,
        d_model: int = 64,
        d_state: int = 16,
        n_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.d_input = d_input
        self.d_model = d_model

        # Input embedding
        self.input_embed = nn.Linear(d_input, d_model)

        # Mamba layers
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state, dropout=dropout)
            for _ in range(n_layers)
        ])

        # Output
        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, d_input)

        # Learnable start token
        self.start_token = nn.Parameter(torch.zeros(1, 1, d_input))

    def forward(self, x):
        """Forward pass (teacher forcing)."""
        h = self.input_embed(x)
        for layer in self.layers:
            h = layer(h)
        h = self.out_norm(h)
        return self.out_proj(h)

    @torch.no_grad()
    def generate(self, batch_size, device='cpu', temperature=1.0):
        """Generate sequences autoregressively."""
        self.eval()

        # Start with learned token
        x = self.start_token.expand(batch_size, -1, -1).to(device)  # (B, 1, d_input)

        for _ in range(self.seq_len - 1):
            # Predict next
            pred = self.forward(x)  # (B, t, d_input)
            next_val = pred[:, -1:, :]  # (B, 1, d_input)

            # Add noise for diversity
            if temperature > 0:
                next_val = next_val + torch.randn_like(next_val) * temperature * 0.1

            x = torch.cat([x, next_val], dim=1)

        return x
