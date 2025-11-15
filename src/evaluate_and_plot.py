#!/usr/bin/env python3
"""
Complete Evaluation and Plotting Script

Loads trained model, evaluates on test set, and generates both required plots.
Supports configurable sequence length L.
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import FreqLSTM
from src.train import get_device, compute_mse

# Parameters
FS = 1000  # Sampling rate [Hz]
T_SEC = 10  # Signal duration [s]
FREQUENCIES = [1, 3, 5, 7]  # Hz
SEQ_LENGTH = 50  # Must match training sequence length


def run_sequence(model, x_seq, device, seq_length=1):
    """
    Run model on a full sequence, maintaining internal state.
    
    Args:
        model: Trained LSTM model
        x_seq: Input sequence of shape (T, 5) - can be numpy or torch tensor
        device: Device to run on
        seq_length: Length of chunks to process at once (L)
    
    Returns:
        Predictions as numpy array of shape (T,)
    """
    model.eval()
    
    # Convert to tensor if needed
    if isinstance(x_seq, np.ndarray):
        x_seq = torch.tensor(x_seq, dtype=torch.float32, device=device)
    
    T = x_seq.shape[0]
    preds = []
    
    with torch.no_grad():
        hidden = model.init_hidden(batch_size=1, device=device)
        
        # Process sequence in chunks of length L
        t = 0
        while t < T:
            chunk_size = min(seq_length, T - t)
            
            x_chunk = x_seq[t:t+chunk_size, :].unsqueeze(0)  # (1, chunk_size, 5)
            y_pred_chunk, hidden = model(x_chunk, hidden)  # (1, chunk_size, 1)
            
            # Extract predictions
            preds.extend(y_pred_chunk.squeeze(0).cpu().numpy())
            
            t += chunk_size
    
    return np.array(preds)


def find_one_example_per_frequency(x):
    """
    Find one example sequence for each frequency.
    
    Args:
        x: Input data of shape (N, T, 5)
    
    Returns:
        Dictionary mapping frequency index to sequence index
    """
    N = x.shape[0]
    idx_for_freq = {}
    
    for i in range(N):
        # Check one-hot encoding at first time step
        C = x[i, 0, 1:5]  # C1..C4
        freq_idx = int(np.argmax(C))
        
        if freq_idx not in idx_for_freq:
            idx_for_freq[freq_idx] = i
        
        # Stop when we have one example for each frequency
        if len(idx_for_freq) == 4:
            break
    
    return idx_for_freq


def plot_single_example(test_x, test_y, model, device, seq_length, project_root):
    """Generate single example extraction plot."""
    # Select first example
    example_idx = 0
    x_seq = test_x[example_idx]  # (T, 5)
    y_true = test_y[example_idx]  # (T,)
    
    # Get predictions
    y_pred = run_sequence(model, x_seq, device, seq_length)
    
    # Extract noisy signal (first column)
    S_noisy = x_seq[:, 0]
    
    # Time vector
    T_total = x_seq.shape[0]
    t = np.linspace(0, T_SEC, T_total, endpoint=False)
    
    # Window for visualization (first 2 seconds)
    window_sec = 2
    window = FS * window_sec
    t_window = t[:window]
    S_window = S_noisy[:window]
    y_true_window = y_true[:window]
    y_pred_window = y_pred[:window]
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(t_window, S_window, label="Noisy mixture S(t)", alpha=0.7, linewidth=1)
    plt.plot(t_window, y_true_window, label="Target clean sinus", linewidth=2)
    plt.plot(t_window, y_pred_window, label="LSTM output", linestyle="--", linewidth=2)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title("Frequency Extraction – Single Test Example")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot to project root
    plots_dir = os.path.join(project_root, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    out_path = os.path.join(plots_dir, "single_example_extraction.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"✓ Plot 1 saved: {out_path}")


def plot_per_frequency(test_x, test_y, model, device, seq_length, project_root):
    """Generate per-frequency extraction plot."""
    # Find one example per frequency
    idx_for_freq = find_one_example_per_frequency(test_x)
    print(f"  Example indices per frequency: {idx_for_freq}")
    
    # Time vector
    T_total = test_x.shape[1]
    t = np.linspace(0, T_SEC, T_total, endpoint=False)
    window_sec = 1
    window = FS * window_sec
    t_window = t[:window]
    
    # Create subplots
    plt.figure(figsize=(12, 8))
    
    for fi in range(4):
        seq_idx = idx_for_freq[fi]
        x_seq = test_x[seq_idx]  # (T, 5)
        y_true = test_y[seq_idx]  # (T,)
        
        # Get predictions
        y_pred = run_sequence(model, x_seq, device, seq_length)
        
        # Window for visualization
        y_true_window = y_true[:window]
        y_pred_window = y_pred[:window]
        
        # Plot in subplot
        plt.subplot(2, 2, fi + 1)
        plt.plot(t_window, y_true_window, label="Target", linewidth=2)
        plt.plot(t_window, y_pred_window, linestyle="--", label="LSTM", linewidth=2)
        plt.title(f"Frequency extraction f = {FREQUENCIES[fi]} Hz")
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.grid(True, alpha=0.3)
        
        # Add legend only to first subplot
        if fi == 0:
            plt.legend()
    
    plt.tight_layout()
    
    # Save plot to project root
    plots_dir = os.path.join(project_root, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    out_path = os.path.join(plots_dir, "per_frequency_extraction.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"✓ Plot 2 saved: {out_path}")


def main():
    """Main evaluation and plotting pipeline."""
    # Get project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Hyperparameters (must match training)
    hidden_size = 64
    seq_length = SEQ_LENGTH  # Must match training sequence length
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    print(f"Sequence length L = {seq_length}\n")
    
    # Load data from project root
    print("Loading data...")
    train_x = np.load(os.path.join(project_root, "train_noisy.npy"))
    train_y = np.load(os.path.join(project_root, "train_target.npy"))
    test_x = np.load(os.path.join(project_root, "test_noisy.npy"))
    test_y = np.load(os.path.join(project_root, "test_target.npy"))
    
    print(f"Train X shape: {train_x.shape}")
    print(f"Train Y shape: {train_y.shape}")
    print(f"Test X shape:  {test_x.shape}")
    print(f"Test Y shape:  {test_y.shape}\n")
    
    # Load model from project root
    print("Loading model...")
    model = FreqLSTM(input_size=5, hidden_size=hidden_size, num_layers=1)
    model_path = os.path.join(project_root, "models", "lstm_freq_extractor.pt")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model loaded from: {model_path}\n")
    
    # Evaluate
    print("Evaluating on datasets...")
    mse_train = compute_mse(model, train_x, train_y, device, seq_length=seq_length)
    mse_test = compute_mse(model, test_x, test_y, device, seq_length=seq_length)
    
    print(f"\n{'='*50}")
    print(f"Evaluation Results:")
    print(f"  Train MSE: {mse_train:.6f}")
    print(f"  Test MSE:  {mse_test:.6f}")
    print(f"{'='*50}\n")
    
    # Generate plots
    print("Generating plots...")
    plot_single_example(test_x, test_y, model, device, seq_length, project_root)
    plot_per_frequency(test_x, test_y, model, device, seq_length, project_root)
    
    print("\n✓ Evaluation and plotting complete!")


if __name__ == "__main__":
    main()

