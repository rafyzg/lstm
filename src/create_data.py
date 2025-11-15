#!/usr/bin/env python3
"""
Data Generation for LSTM Frequency Extraction Assignment

Generates:
- Noisy mixed signals S(t) from 4 sinusoids (1, 3, 5, 7 Hz)
- Clean target signals (sinusoids without noise)
- Input sequences: [S(t), C1, C2, C3, C4] where C is one-hot frequency selector
"""

import os
import sys
import numpy as np

# Add parent directory to path for data saving
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ============================================================
# PARAMETERS
# ============================================================
FS = 1000          # Sampling rate [Hz]
T_SEC = 10         # Signal duration [s]
N_SAMPLES = FS * T_SEC  # Total samples: 10000

FREQUENCIES = np.array([1, 3, 5, 7], dtype=float)  # f1, f2, f3, f4 [Hz]


def generate_dataset(seed: int, n_mixtures: int):
    """
    Generate dataset for LSTM frequency extraction.
    
    For each mixture:
    - Creates noisy signal S(t) = average of 4 sinusoids with random A and phi
    - For each frequency, creates a sequence with:
        - Input: [S(t), C1, C2, C3, C4] where C is one-hot for the frequency
        - Target: clean sin(2*pi*f*t) for that frequency
    
    Args:
        seed: Random seed for reproducibility
        n_mixtures: Number of different noisy mixtures to generate
        
    Returns:
        X: Input sequences, shape (n_mixtures * 4, N_SAMPLES, 5)
        Y: Target sequences, shape (n_mixtures * 4, N_SAMPLES)
    """
    np.random.seed(seed)
    
    # Time vector
    t = np.linspace(0, T_SEC, N_SAMPLES, endpoint=False)
    
    X_list = []  # Will store input sequences
    Y_list = []  # Will store target sequences
    
    for _ in range(n_mixtures):
        # Generate noisy mixture S(t)
        S = np.zeros(N_SAMPLES)
        
        for f in FREQUENCIES:
            # Random amplitude and phase for each frequency
            A = np.random.uniform(0.8, 1.2)
            phi = np.random.uniform(0, 2 * np.pi)
            S += A * np.sin(2 * np.pi * f * t + phi)
        
        # Average the 4 sinusoids
        S /= 4.0
        
        # Generate clean targets for all frequencies
        targets = np.array([
            np.sin(2 * np.pi * f * t) for f in FREQUENCIES
        ])  # Shape: (4, N_SAMPLES)
        
        # Create 4 sequences per mixture, one for each frequency
        for freq_idx in range(4):
            # One-hot encoding for frequency selection
            C = np.zeros(4, dtype=float)
            C[freq_idx] = 1.0
            
            # Repeat one-hot vector for all time steps
            C_matrix = np.tile(C, (N_SAMPLES, 1))  # Shape: (N_SAMPLES, 4)
            S_vector = S.reshape(-1, 1)            # Shape: (N_SAMPLES, 1)
            
            # Concatenate: [S(t), C1, C2, C3, C4]
            X_seq = np.hstack([S_vector, C_matrix])  # Shape: (N_SAMPLES, 5)
            Y_seq = targets[freq_idx]                 # Shape: (N_SAMPLES,)
            
            X_list.append(X_seq)
            Y_list.append(Y_seq)
    
    # Stack all sequences
    X = np.stack(X_list, axis=0)  # Shape: (n_mixtures * 4, N_SAMPLES, 5)
    Y = np.stack(Y_list, axis=0)  # Shape: (n_mixtures * 4, N_SAMPLES)
    
    return X, Y


def main():
    """Generate and save train and test datasets."""
    # Get project root directory (parent of src)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Hyperparameters
    n_mixtures_train = 100  # Each gives 4 sequences => 400 training sequences
    n_mixtures_test = 20     # Each gives 4 sequences => 80 test sequences
    
    print("Generating training data...")
    train_x, train_y = generate_dataset(seed=1, n_mixtures=n_mixtures_train)
    
    print("Generating test data...")
    test_x, test_y = generate_dataset(seed=2, n_mixtures=n_mixtures_test)
    
    # Print shapes
    print(f"\nTrain X shape: {train_x.shape}")  # Expected: (400, 10000, 5)
    print(f"Train Y shape: {train_y.shape}")    # Expected: (400, 10000)
    print(f"Test X shape:  {test_x.shape}")    # Expected: (80, 10000, 5)
    print(f"Test Y shape:  {test_y.shape}")     # Expected: (80, 10000)
    
    # Save data to project root
    print("\nSaving data...")
    np.save(os.path.join(project_root, "train_noisy.npy"), train_x)
    np.save(os.path.join(project_root, "train_target.npy"), train_y)
    np.save(os.path.join(project_root, "test_noisy.npy"), test_x)
    np.save(os.path.join(project_root, "test_target.npy"), test_y)
    
    print("Data saved successfully!")
    print("  - train_noisy.npy: Training input sequences")
    print("  - train_target.npy: Training target sequences")
    print("  - test_noisy.npy: Test input sequences")
    print("  - test_target.npy: Test target sequences")


if __name__ == "__main__":
    main()

