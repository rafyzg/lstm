# Hyperparameter Tuning Guide

## Problem: High MSE (~0.5)

The initial model configuration resulted in MSE values around 0.5, which indicates the model is not learning effectively. This document explains the optimizations applied.

## Changes Made

### 1. Increased Hidden Size: 32 → 64
**Why**: 
- Larger hidden size provides more capacity to learn complex frequency extraction patterns
- 32 was too small for the task complexity
- 64 is a good balance between capacity and training speed

### 2. Increased Epochs: 10 → 50
**Why**:
- 10 epochs was insufficient for convergence
- The model needs more time to learn the frequency extraction task
- With proper learning rate scheduling, 50 epochs allows full convergence

### 3. Reduced Learning Rate: 1e-3 → 5e-4
**Why**:
- Lower learning rate provides more stable training
- Prevents overshooting optimal weights
- Combined with scheduler, allows fine-tuning in later epochs

### 4. Increased Sequence Length: L=10 → L=50
**Why**:
- Longer sequences provide better gradient flow
- Model can see more context at once
- Better parallelization and faster training
- Still maintains hidden state between chunks

### 5. Added Learning Rate Scheduler
**Why**:
- `ReduceLROnPlateau` reduces learning rate when loss plateaus
- Helps fine-tune in later training stages
- Prevents getting stuck in local minima

### 6. Added Gradient Clipping
**Why**:
- Prevents exploding gradients that can destabilize training
- Clips gradients to max_norm=1.0
- Especially important for RNNs/LSTMs

## New Configuration

```python
hidden_size = 64
num_epochs = 50
lr = 5e-4
seq_length = 50
```

## Expected Results

With these changes, you should see:
- **Initial loss**: ~0.3-0.4 (starts lower due to better initialization)
- **Final Train MSE**: < 0.01 (ideally 0.001-0.005)
- **Final Test MSE**: < 0.01 (similar to train, showing good generalization)
- **Convergence**: Should see steady decrease in loss over epochs

## Alternative Configurations

If the above doesn't work well, try these alternatives:

### Option 1: Serial Learning (L=1)
```python
seq_length = 1
num_epochs = 30
lr = 1e-3
hidden_size = 64
```
- Most stable, follows assignment specification exactly
- Slower but often more reliable
- Better for debugging

### Option 2: Very Long Sequences (L=100)
```python
seq_length = 100
num_epochs = 30
lr = 3e-4
hidden_size = 64
```
- Fastest training
- Requires more memory
- Good for final training after debugging

### Option 3: Larger Model
```python
hidden_size = 128
num_epochs = 40
lr = 3e-4
seq_length = 50
```
- More capacity for complex patterns
- Slower training
- Use if 64 hidden size isn't enough

## Monitoring Training

Watch for these signs:

1. **Good Training**:
   - Loss decreases steadily
   - Learning rate decreases when loss plateaus
   - Final loss < 0.01

2. **Poor Training**:
   - Loss stays constant or increases
   - Loss oscillates wildly
   - Final loss > 0.1

3. **Overfitting**:
   - Train MSE << Test MSE (gap > 0.01)
   - Solution: Reduce epochs or add regularization

## Troubleshooting

### If MSE is still high (> 0.1):
1. Check data generation - ensure random seeds are correct
2. Try L=1 (serial learning) first to verify model works
3. Increase hidden size to 128
4. Try different learning rates: 1e-4, 3e-4, 1e-3
5. Check if model is actually updating (print gradients)

### If training is too slow:
1. Increase seq_length (L=100)
2. Reduce hidden size temporarily for testing
3. Use fewer epochs initially to test configuration

### If loss decreases but MSE is still high:
1. Model might need more epochs
2. Check evaluation code - ensure seq_length matches
3. Verify model is in eval mode during evaluation

## Next Steps

1. Run training with new hyperparameters:
   ```bash
   python src/train.py
   ```

2. Monitor the loss - it should decrease steadily

3. After training, evaluate:
   ```bash
   python src/evaluate_and_plot.py
   ```

4. If results are still poor, try the alternative configurations above

