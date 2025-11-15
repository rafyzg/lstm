#!/usr/bin/env python3
"""
LSTM Model for Frequency Extraction

Takes input [S(t), C1, C2, C3, C4] and outputs the clean sinusoid
for the frequency specified by the one-hot encoding C.
"""

import torch
import torch.nn as nn


class FreqLSTM(nn.Module):
    """
    LSTM model for extracting specific frequencies from noisy mixtures.
    
    Input: [S(t), C1, C2, C3, C4] where:
        - S(t): noisy mixed signal
        - C1..C4: one-hot encoding for frequency selection
    
    Output: Clean sinusoid for the selected frequency
    """
    
    def __init__(self, input_size=5, hidden_size=32, num_layers=1):
        super(FreqLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer: processes input sequence
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True  # Input shape: (batch, seq_len, input_size)
        )
        
        # Linear layer: maps hidden state to scalar output
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x, hidden):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_size=5)
               where seq_len can be 1 (serial learning) or >1 (chunked processing)
            hidden: Tuple (h, c) where:
                h: (num_layers, batch, hidden_size)
                c: (num_layers, batch, hidden_size)
        
        Returns:
            out: Output tensor of shape (batch, seq_len, 1)
            hidden: Updated hidden state tuple
        """
        out, hidden = self.lstm(x, hidden)  # out: (batch, seq_len, hidden_size)
        out = self.fc(out)                  # out: (batch, seq_len, 1)
        return out, hidden
    
    def init_hidden(self, batch_size=1, device="cpu"):
        """
        Initialize hidden state to zeros.
        
        Args:
            batch_size: Batch size (default 1 for serial learning)
            device: Device to create tensors on
        
        Returns:
            Tuple (h0, c0) of zero-initialized hidden states
        """
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return (h0, c0)

