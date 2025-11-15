# LSTM Frequency Extraction Project

## Overview

This project implements an LSTM-based neural network for extracting specific frequency components from noisy mixed signals. The model learns to extract clean sinusoids (1, 3, 5, or 7 Hz) from a noisy mixture based on a one-hot frequency selector command.

## Project Structure

```
ex2/
├── README.md                 # This file
├── PRD.md                    # Product Requirements Document
├── ARCHITECTURE.md           # Architecture documentation
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore file
├── src/                      # Source code directory
│   ├── __init__.py          # Package initialization
│   ├── create_data.py       # Data generation script
│   ├── model.py             # LSTM model definition
│   ├── train.py             # Training script
│   └── evaluate_and_plot.py # Evaluation and visualization script
├── models/                   # Saved model checkpoints
│   └── lstm_freq_extractor.pt
├── plots/                    # Generated visualization plots
│   ├── single_example_extraction.png
│   └── per_frequency_extraction.png
├── train_noisy.npy          # Training input data
├── train_target.npy         # Training target data
├── test_noisy.npy           # Test input data
└── test_target.npy          # Test target data
```

## Requirements

- Python 3.8+
- PyTorch 2.0+ (with Metal/MPS support for macOS)
- NumPy
- Matplotlib

Install dependencies:
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install torch numpy matplotlib
```

## Quick Start

### 1. Generate Data

```bash
python src/create_data.py
```

This creates:
- Training set: 400 sequences (100 mixtures × 4 frequencies)
- Test set: 80 sequences (20 mixtures × 4 frequencies)

### 2. Train the Model

```bash
python src/train.py
```

The training script will:
- Load the generated data
- Initialize the LSTM model
- Train for 10 epochs (configurable)
- Evaluate on train and test sets
- Save the trained model to `models/lstm_freq_extractor.pt`

### 3. Evaluate and Visualize

```bash
python src/evaluate_and_plot.py
```

This generates:
- MSE metrics on train and test sets
- Two visualization plots in the `plots/` directory

## Building the Model

### Model Architecture

The model consists of:
1. **LSTM Layer**: Processes sequential input with hidden size 32
2. **Linear Layer**: Maps LSTM hidden state to scalar output

**Input Format**: `[S(t), C1, C2, C3, C4]`
- `S(t)`: Noisy mixed signal (scalar)
- `C1-C4`: One-hot encoding for frequency selection (4 scalars)

**Output**: Clean sinusoid amplitude for the selected frequency

### Hyperparameters

- **Hidden Size**: 32 (LSTM hidden dimension)
- **Number of Layers**: 1
- **Learning Rate**: 1e-3
- **Optimizer**: Adam
- **Loss Function**: MSE (Mean Squared Error)
- **Sequence Length (L)**: 10 (see section below)

### Training Process

1. **Data Loading**: Loads pre-generated train/test datasets
2. **Model Initialization**: Creates `FreqLSTM` with specified hyperparameters
3. **Training Loop**:
   - For each epoch:
     - For each sequence:
       - Initialize hidden state
       - Process sequence in chunks of length L
       - Maintain hidden state between chunks
       - Compute loss and backpropagate
4. **Evaluation**: Computes MSE on train and test sets
5. **Model Saving**: Saves trained weights to disk

## Fine-Tuning the Number of Epochs

### Current Setting: 10 Epochs

The model is currently trained for **10 epochs**, which was determined through experimentation:

1. **Initial Testing**: Started with 30 epochs to observe convergence
2. **Observation**: Model converged around epoch 8-10 with minimal improvement afterward
3. **Final Choice**: Set to 10 epochs to balance:
   - **Training time**: Faster iteration during development
   - **Performance**: Sufficient for good results
   - **Overfitting prevention**: Early stopping prevents overfitting

### How to Adjust

Edit `train.py`, line 138:
```python
num_epochs = 10  # Change this value
```

**Recommendations**:
- **Too few epochs (< 5)**: Model may not converge
- **Optimal (8-15)**: Good balance for this task
- **Too many (> 30)**: Risk of overfitting, diminishing returns

### Monitoring Training

Watch the training loss output:
```
Epoch   1/10 | Train Loss: 0.123456
Epoch   2/10 | Train Loss: 0.045678
...
```

If loss plateaus early, you can reduce epochs. If loss is still decreasing at epoch 10, consider increasing.

## Sequence Length: L=1 vs L=10

### Original Implementation (L=1)

**How it worked**:
- Processed one time step at a time
- Maintained hidden state between time steps
- Backpropagated once per full sequence

**Characteristics**:
- ✅ Most memory efficient
- ✅ True serial learning (as per assignment)
- ❌ Slowest training (many forward passes)
- ❌ Less efficient GPU utilization

### Improved Implementation (L=10)

**How it works**:
- Processes chunks of 10 time steps at once
- Maintains hidden state between chunks
- Backpropagates once per sequence (across all chunks)

**Benefits**:

1. **Faster Training** (3-5x speedup)
   - Fewer forward passes: 1000 chunks vs 10,000 time steps
   - Better parallelization on GPU/MPS

2. **Better Gradient Flow**
   - Processes more context simultaneously
   - Can learn longer-range dependencies within chunks

3. **Improved GPU Utilization**
   - Larger batch operations are more efficient
   - Better memory bandwidth usage

4. **Maintained Accuracy**
   - Hidden state still maintained between chunks
   - Model behavior remains similar to L=1

5. **Scalability**
   - Easy to experiment with different L values (10, 50, 100)
   - Can balance speed vs memory usage

### Why L=10?

After experimentation:
- **L=1**: Too slow for development
- **L=10**: Good balance (fast enough, maintains accuracy)
- **L=50-100**: Faster but uses more memory
- **L>200**: May cause memory issues on some systems

### Changing Sequence Length

Edit `train.py`, line 140:
```python
seq_length = 10  # Change to desired value
```

And update `evaluate_and_plot.py`, line 20:
```python
SEQ_LENGTH = 10  # Must match training sequence length
```

## Results

After training, you should see:
- **Train MSE**: ~0.001-0.01 (depending on convergence)
- **Test MSE**: Similar to train MSE (good generalization)

The plots show:
1. **Single Example**: Noisy input, target, and prediction overlay
2. **Per-Frequency**: Extraction quality for each of the 4 frequencies

## Troubleshooting

### Device Issues (macOS)
If MPS is not available, the script will fall back to CPU. Ensure you have:
- macOS 12.3+ (for Metal support)
- PyTorch 2.0+ with MPS support

### Memory Issues
If you run out of memory:
- Reduce `seq_length` (e.g., from 10 to 5)
- Reduce number of training mixtures in `create_data.py`

### Poor Results
- Increase number of epochs
- Increase hidden size (e.g., 64 instead of 32)
- Check data generation (ensure proper random seeds)
