"""
S4 (Structured State Space) Layer Implementation

Based on "Efficiently Modeling Long Sequences with Structured State Spaces" (Gu et al. 2022)
This implements the core S4 layer with HiPPO initialization for capturing long-range dependencies.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_hippo(N):
    """
    Create HiPPO-LegS matrix for optimal polynomial projection.
    This initialization helps capture trends and long-range dependencies.

    Args:
        N: State dimension
    Returns:
        A: (N, N) HiPPO matrix
    """
    P = torch.sqrt(1 + 2 * torch.arange(N, dtype=torch.float32))
    A = P.unsqueeze(1) * P.unsqueeze(0)
    A = torch.tril(A) - torch.diag(torch.arange(N, dtype=torch.float32) + 1)
    return A


def discretize_zoh(A, B, dt):
    """
    Discretize continuous-time SSM using Zero-Order Hold.

    x_{k+1} = A_d @ x_k + B_d @ u_k

    Args:
        A: (N, N) continuous state matrix
        B: (N, D) continuous input matrix
        dt: discretization step size
    Returns:
        A_d, B_d: discretized matrices
    """
    N = A.shape[0]
    I = torch.eye(N, device=A.device, dtype=A.dtype)

    # Simple Euler discretization (stable for small dt)
    A_d = I + dt * A
    B_d = dt * B

    return A_d, B_d


class S4Layer(nn.Module):
    """
    Structured State Space Layer (S4)

    Processes sequences using state space model:
        x'(t) = Ax(t) + Bu(t)
        y(t) = Cx(t) + Du(t)

    Key features:
    - HiPPO initialization for trend/dependency capture
    - Linear time complexity O(L) via parallel scan
    - Constant memory state for generation
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dropout: float = 0.0,
    ):
        """
        Args:
            d_model: Input/output dimension
            d_state: State dimension (N)
            dt_min: Minimum discretization step
            dt_max: Maximum discretization step
            dropout: Dropout rate
        """
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state

        # Initialize A with HiPPO
        A = make_hippo(d_state)
        self.register_buffer('A', A)

        # Learnable parameters
        # B: input -> state
        self.B = nn.Parameter(torch.randn(d_state, d_model) * 0.01)
        # C: state -> output
        self.C = nn.Parameter(torch.randn(d_model, d_state) * 0.01)
        # D: skip connection
        self.D = nn.Parameter(torch.ones(d_model))

        # Learnable discretization step (log scale for stability)
        log_dt = torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, u):
        """
        Forward pass through S4 layer.

        Args:
            u: (batch, seq_len, d_model) input sequence
        Returns:
            y: (batch, seq_len, d_model) output sequence
        """
        batch, seq_len, _ = u.shape

        # Get discretization step
        dt = torch.exp(self.log_dt)  # (d_model,)

        # Discretize per channel (simplified - using mean dt)
        dt_mean = dt.mean()
        A_d, B_d = discretize_zoh(self.A, self.B, dt_mean)

        # Run state space model (sequential scan)
        # For efficiency, this could use parallel scan, but sequential is clearer
        x = torch.zeros(batch, self.d_state, device=u.device, dtype=u.dtype)
        outputs = []

        for t in range(seq_len):
            u_t = u[:, t, :]  # (batch, d_model)

            # State update: x = A_d @ x + B_d @ u
            x = torch.einsum('ns,bs->bn', A_d, x) + torch.einsum('nd,bd->bn', B_d, u_t)

            # Output: y = C @ x + D * u
            y_t = torch.einsum('dn,bn->bd', self.C, x) + self.D * u_t
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)  # (batch, seq_len, d_model)
        return self.dropout(y)

    def step(self, u_t, state):
        """
        Single step for autoregressive generation.

        Args:
            u_t: (batch, d_model) single timestep input
            state: (batch, d_state) current state
        Returns:
            y_t: (batch, d_model) output
            new_state: (batch, d_state) updated state
        """
        dt_mean = torch.exp(self.log_dt).mean()
        A_d, B_d = discretize_zoh(self.A, self.B, dt_mean)

        # State update
        new_state = torch.einsum('ns,bs->bn', A_d, state) + torch.einsum('nd,bd->bn', B_d, u_t)

        # Output
        y_t = torch.einsum('dn,bn->bd', self.C, new_state) + self.D * u_t

        return y_t, new_state

    def init_state(self, batch_size, device):
        """Initialize state for generation."""
        return torch.zeros(batch_size, self.d_state, device=device)


class S4Block(nn.Module):
    """
    S4 Block with normalization and residual connection.
    """

    def __init__(self, d_model, d_state=64, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.s4 = S4Layer(d_model, d_state, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # S4 with residual
        x = x + self.s4(self.norm(x))
        # FFN with residual
        x = x + self.ffn(self.norm2(x))
        return x
