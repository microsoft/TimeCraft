"""
Train Mamba-based SSM on oil field time series data.

This script trains the MambaVAE model on the oilfield dataset
and evaluates its ability to capture pattern-specific characteristics.
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from SSM.mamba_tsg import MambaVAE


def train_mamba_oilfield():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seq_len', type=int, default=168)
    parser.add_argument('--d_input', type=int, default=1)
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--d_state', type=int, default=16)
    parser.add_argument('--d_latent', type=int, default=32)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--kl_weight', type=float, default=0.1)
    parser.add_argument('--dropout', type=float, default=0.1)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load oilfield training data
    data_path = '/Users/jameslepage/Projects/TimeCraft/data/oilfield/oilfield_168_train.npy'
    data = np.load(data_path)
    print(f"Loaded data shape: {data.shape}")

    # Reshape to (N, seq_len, channels)
    if data.ndim == 2:
        data = data[:, :, np.newaxis]
    elif data.ndim == 3 and data.shape[1] == 1:
        data = data.transpose(0, 2, 1)

    # Normalize
    data_mean = data.mean()
    data_std = data.std()
    data_norm = (data - data_mean) / (data_std + 1e-8)

    print(f"Normalized data shape: {data_norm.shape}")
    print(f"Data mean: {data_mean:.4f}, std: {data_std:.4f}")

    # Create dataset
    tensor_data = torch.from_numpy(data_norm).float()
    dataset = Data.TensorDataset(tensor_data)
    loader = Data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Create model
    model = MambaVAE(
        seq_len=args.seq_len,
        d_input=args.d_input,
        d_model=args.d_model,
        d_state=args.d_state,
        d_latent=args.d_latent,
        n_layers=args.n_layers,
        dropout=args.dropout,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nMambaVAE parameters: {n_params:,}")
    print(f"Training for {args.epochs} epochs...")
    print("=" * 50)

    # Training loop
    best_loss = float('inf')
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        total_rec = 0
        total_kl = 0

        for batch in loader:
            x = batch[0].to(device)

            optimizer.zero_grad()
            x_rec, mu, logvar = model(x)

            loss, rec_loss, kl_loss = model.loss(x, x_rec, mu, logvar, args.kl_weight)
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            total_loss += loss.item()
            total_rec += rec_loss.item()
            total_kl += kl_loss.item()

        scheduler.step()
        n_batches = len(loader)
        avg_loss = total_loss / n_batches

        if avg_loss < best_loss:
            best_loss = avg_loss

        if (epoch + 1) % 25 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d} | Loss: {avg_loss:.4f} | Rec: {total_rec/n_batches:.4f} | KL: {total_kl/n_batches:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

    print("=" * 50)
    print(f"Training complete! Best loss: {best_loss:.4f}")

    # Save model
    output_dir = '/Users/jameslepage/Projects/TimeCraft/oilfield/mamba_output'
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), f'{output_dir}/mamba_oilfield.pth')
    print(f"Model saved to {output_dir}/mamba_oilfield.pth")

    # Generate samples
    print("\nGenerating samples...")
    model.eval()

    patterns = ['normal', 'degradation', 'anomaly', 'vibration', 'pressure']

    with torch.no_grad():
        for pattern in patterns:
            samples = model.sample(20, device=device).cpu().numpy()
            # Denormalize
            samples = samples * data_std + data_mean

            out_path = f'{output_dir}/mamba_{pattern}_generated.npy'
            np.save(out_path, samples)
            print(f"  {pattern}: saved {samples.shape}")

    print(f"\nAll outputs saved to: {output_dir}")

    # Quality analysis
    print("\n" + "=" * 50)
    print("QUALITY ANALYSIS")
    print("=" * 50)

    def calc_slopes(data):
        data = data.squeeze()
        return [np.polyfit(np.arange(len(s)), s, 1)[0] for s in data]

    def calc_zero_crossings(data):
        data = data.squeeze()
        return [np.sum(np.diff(np.sign(s - s.mean())) != 0) for s in data]

    def calc_periodicity(data):
        data = data.squeeze()
        periods = []
        for s in data:
            s_centered = s - s.mean()
            corr = np.correlate(s_centered, s_centered, mode='full')
            corr = corr[len(corr)//2:]
            peaks = np.where((corr[1:-1] > corr[:-2]) & (corr[1:-1] > corr[2:]))[0] + 1
            periods.append(peaks[0] if len(peaks) > 0 else 0)
        return periods

    # Compare against original patterns
    for pattern in ['degradation', 'vibration', 'pressure']:
        orig_path = f'/Users/jameslepage/Projects/TimeCraft/data/oilfield/oilfield_{pattern}_168_samples.npy'
        gen_path = f'{output_dir}/mamba_{pattern}_generated.npy'

        orig = np.load(orig_path)
        gen = np.load(gen_path)

        print(f"\n{pattern.upper()}:")

        if pattern == 'degradation':
            orig_slopes = calc_slopes(orig)
            gen_slopes = calc_slopes(gen)
            print(f"  Trend slope - Orig: {np.mean(orig_slopes):.4f}, Gen: {np.mean(gen_slopes):.4f}")
            print(f"  % negative  - Orig: {np.mean([s<0 for s in orig_slopes])*100:.0f}%, Gen: {np.mean([s<0 for s in gen_slopes])*100:.0f}%")

        elif pattern == 'vibration':
            orig_zc = calc_zero_crossings(orig)
            gen_zc = calc_zero_crossings(gen)
            print(f"  Zero crossings - Orig: {np.mean(orig_zc):.1f}, Gen: {np.mean(gen_zc):.1f}")

        elif pattern == 'pressure':
            orig_per = calc_periodicity(orig)
            gen_per = calc_periodicity(gen)
            print(f"  Period length - Orig: {np.mean(orig_per):.1f}, Gen: {np.mean(gen_per):.1f}")


if __name__ == '__main__':
    train_mamba_oilfield()
