#!/usr/bin/env python3
"""
Training Script for LSTM Frequency Extraction

Supports both serial learning (L=1) and batch sequence learning (L>1).
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import FreqLSTM


def get_device():
    """Get the best available device (Metal/MPS for macOS, else CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def train_model(model, train_x, train_y, device, num_epochs=30, lr=1e-3, seq_length=1):
    """
    Train the LSTM model.
    
    Args:
        seq_length: Length of sequences to process at once (L). 
                   L=1 means serial learning (one time step at a time).
                   L>1 means processing L time steps in parallel.
    """
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Add learning rate scheduler for better convergence
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Convert to tensors and move to device
    train_x_tensor = torch.tensor(train_x, dtype=torch.float32, device=device)
    train_y_tensor = torch.tensor(train_y, dtype=torch.float32, device=device)
    
    N_seq, T, _ = train_x_tensor.shape
    
    print(f"Training on {N_seq} sequences, {T} time steps each")
    print(f"Sequence length L = {seq_length}")
    print(f"Using device: {device}\n")
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        # Loop over all sequences
        for i in range(N_seq):
            # Initialize hidden state at the start of each sequence
            hidden = model.init_hidden(batch_size=1, device=device)
            optimizer.zero_grad()
            total_loss = 0.0
            num_chunks = 0
            
            # Process sequence in chunks of length L
            t = 0
            while t < T:
                # Determine chunk size (last chunk may be shorter)
                chunk_size = min(seq_length, T - t)
                
                # Get chunk of input and target
                x_chunk = train_x_tensor[i, t:t+chunk_size, :].unsqueeze(0)  # (1, chunk_size, 5)
                y_chunk = train_y_tensor[i, t:t+chunk_size].unsqueeze(0).unsqueeze(-1)  # (1, chunk_size, 1)
                
                # Forward pass: process entire chunk at once
                y_pred_chunk, hidden = model(x_chunk, hidden)
                
                # Compute loss for this chunk
                chunk_loss = criterion(y_pred_chunk, y_chunk)
                total_loss = total_loss + chunk_loss
                num_chunks += 1
                
                # Move to next chunk
                t += chunk_size
            
            # Backpropagate once per sequence
            total_loss.backward()
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Average loss per time step
            epoch_loss += total_loss.item() / T
        
        # Average loss over all sequences
        epoch_loss /= N_seq
        # Update learning rate based on loss
        scheduler.step(epoch_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:3d}/{num_epochs} | Train Loss: {epoch_loss:.6f} | LR: {current_lr:.2e}")


def compute_mse(model, x, y, device, seq_length=1):
    """
    Compute Mean Squared Error on a dataset.
    
    Args:
        seq_length: Length of sequences to process at once (L)
    
    Returns:
        MSE averaged over all time steps and sequences
    """
    model.eval()
    x_tensor = torch.tensor(x, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y, dtype=torch.float32, device=device)
    
    N_seq, T, _ = x_tensor.shape
    mse_sum = 0.0
    count = 0
    
    with torch.no_grad():
        for i in range(N_seq):
            hidden = model.init_hidden(batch_size=1, device=device)
            
            # Process sequence in chunks of length L
            t = 0
            while t < T:
                chunk_size = min(seq_length, T - t)
                
                x_chunk = x_tensor[i, t:t+chunk_size, :].unsqueeze(0)  # (1, chunk_size, 5)
                y_chunk = y_tensor[i, t:t+chunk_size].unsqueeze(0).unsqueeze(-1)  # (1, chunk_size, 1)
                
                y_pred_chunk, hidden = model(x_chunk, hidden)
                
                # Accumulate MSE
                mse_sum += ((y_pred_chunk - y_chunk) ** 2).sum().item()
                count += chunk_size
                
                t += chunk_size
    
    return mse_sum / count


def main():
    """Main training pipeline."""
    # Get project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Hyperparameters - Optimized for better convergence
    hidden_size = 64  # Increased from 32 for more capacity
    num_epochs = 50   # Increased from 10 for better convergence
    lr = 5e-4         # Reduced from 1e-3 for more stable training
    seq_length = 50   # Increased from 10 for better gradient flow
    
    # Alternative configurations to try:
    # Option 1: Serial learning (L=1) - slower but often more stable
    # seq_length = 1
    # num_epochs = 30
    # lr = 1e-3
    
    # Option 2: Very long sequences (L=100) - faster but needs more memory
    # seq_length = 100
    # num_epochs = 30
    # lr = 3e-4
    
    # Get device (Metal/MPS for macOS)
    device = get_device()
    print(f"Using device: {device}\n")
    
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
    
    # Create model
    model = FreqLSTM(input_size=5, hidden_size=hidden_size, num_layers=1)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters\n")
    
    # Train
    print("Starting training...")
    train_model(model, train_x, train_y, device, num_epochs=num_epochs, lr=lr, seq_length=seq_length)
    
    # Evaluate
    print("\nEvaluating...")
    mse_train = compute_mse(model, train_x, train_y, device, seq_length=seq_length)
    mse_test = compute_mse(model, test_x, test_y, device, seq_length=seq_length)
    
    print(f"\n{'='*50}")
    print(f"Final Results:")
    print(f"  Train MSE: {mse_train:.6f}")
    print(f"  Test MSE:  {mse_test:.6f}")
    print(f"{'='*50}\n")
    
    # Save model to project root
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "lstm_freq_extractor.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()

