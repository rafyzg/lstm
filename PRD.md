# Product Requirements Document (PRD)
## LSTM Frequency Extraction System

### 1. Project Overview

**Project Name**: LSTM Frequency Extraction from Noisy Signals

**Version**: 1.0

**Date**: 2024

**Purpose**: Develop and train an LSTM neural network capable of extracting specific frequency components from noisy mixed signals based on a frequency selector command.

### 2. Objectives

#### 2.1 Primary Objectives
- Extract clean sinusoids (1, 3, 5, or 7 Hz) from noisy mixed signals
- Learn frequency extraction based on one-hot encoded frequency commands
- Achieve low MSE (< 0.01) on both training and test sets
- Generate visualization plots demonstrating extraction quality

#### 2.2 Secondary Objectives
- Support configurable sequence length (L) for training efficiency
- Optimize for macOS Metal (MPS) acceleration
- Maintain clean, well-documented codebase

### 3. Functional Requirements

#### 3.1 Data Generation (FR-1)
- **Requirement**: Generate synthetic training and test datasets
- **Details**:
  - Create noisy mixed signals from 4 sinusoids (1, 3, 5, 7 Hz)
  - Each sinusoid has random amplitude (0.8-1.2) and phase (0-2π)
  - Generate clean target signals (no noise, no random phase/amplitude)
  - Create input sequences: `[S(t), C1, C2, C3, C4]` where C is one-hot frequency selector
  - Training set: 400 sequences (100 mixtures × 4 frequencies)
  - Test set: 80 sequences (20 mixtures × 4 frequencies)

#### 3.2 Model Architecture (FR-2)
- **Requirement**: Implement LSTM-based frequency extractor
- **Details**:
  - Input size: 5 (1 signal + 4 one-hot commands)
  - LSTM hidden size: 32
  - LSTM layers: 1
  - Output: Scalar (sinusoid amplitude)
  - Linear layer to map hidden state to output

#### 3.3 Training (FR-3)
- **Requirement**: Train model using serial learning with configurable sequence length
- **Details**:
  - Support sequence length L (default: 10)
  - Process sequences in chunks of length L
  - Maintain hidden state between chunks
  - Use Adam optimizer with learning rate 1e-3
  - Train for 10 epochs (configurable)
  - Use MSE loss function

#### 3.4 Evaluation (FR-4)
- **Requirement**: Evaluate model performance
- **Details**:
  - Compute MSE on training set
  - Compute MSE on test set
  - Generate predictions for visualization

#### 3.5 Visualization (FR-5)
- **Requirement**: Generate visualization plots
- **Details**:
  - Plot 1: Single example showing noisy input, target, and prediction
  - Plot 2: Per-frequency extraction (2×2 subplots for 4 frequencies)
  - Save plots as PNG files (150 DPI)

### 4. Non-Functional Requirements

#### 4.1 Performance (NFR-1)
- Training time: < 5 minutes on MPS (MacBook with Metal)
- Inference time: < 1 second per sequence
- Memory usage: < 2 GB during training

#### 4.2 Compatibility (NFR-2)
- Python 3.8+
- PyTorch 2.0+ with MPS support
- macOS 12.3+ for Metal acceleration
- Fallback to CPU if MPS unavailable

#### 4.3 Code Quality (NFR-3)
- Clean, readable code with docstrings
- Modular design (separate files for model, training, evaluation)
- Type hints where appropriate
- Consistent naming conventions

#### 4.4 Documentation (NFR-4)
- README with setup and usage instructions
- Architecture documentation
- Code comments explaining key decisions
- PRD (this document)

### 5. Success Criteria

#### 5.1 Performance Metrics
- **Train MSE**: < 0.01
- **Test MSE**: < 0.01 (within 20% of train MSE)
- **Visual Quality**: Predictions visually match targets in plots

#### 5.2 Code Quality Metrics
- All functions have docstrings
- Code follows PEP 8 style guide
- Project structure is clear and organized

### 6. Constraints

#### 6.1 Technical Constraints
- Must use PyTorch framework
- Must support macOS Metal (MPS)
- Sequence length L must be configurable
- Model must maintain hidden state across chunks

#### 6.2 Data Constraints
- Fixed frequencies: 1, 3, 5, 7 Hz
- Fixed sampling rate: 1000 Hz
- Fixed signal duration: 10 seconds
- Random seed for reproducibility

### 7. Dependencies

- **PyTorch**: Neural network framework
- **NumPy**: Numerical computations
- **Matplotlib**: Visualization

### 8. Deliverables

1. ✅ Data generation script (`create_data.py`)
2. ✅ Model definition (`model.py`)
3. ✅ Training script (`train.py`)
4. ✅ Evaluation and plotting script (`evaluate_and_plot.py`)
5. ✅ Trained model checkpoint (`models/lstm_freq_extractor.pt`)
6. ✅ Visualization plots (`plots/*.png`)
7. ✅ Documentation (README, PRD, ARCHITECTURE)

### 9. Timeline

- **Phase 1**: Data generation and model design (Completed)
- **Phase 2**: Training implementation (Completed)
- **Phase 3**: Evaluation and visualization (Completed)
- **Phase 4**: Documentation and optimization (Current)

### 10. Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Poor model convergence | High | Experiment with hyperparameters, ensure proper data generation |
| MPS compatibility issues | Medium | Fallback to CPU, test on multiple systems |
| Memory limitations | Low | Reduce sequence length or batch size |
| Overfitting | Medium | Monitor train/test MSE gap, early stopping |
