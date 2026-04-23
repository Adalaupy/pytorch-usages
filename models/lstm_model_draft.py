"""
Long Short-Term Memory (LSTM) Network:

- Sequence classification
- Time series prediction
- Sequential data analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM_Model(nn.Module):
    
    def __init__(self, input_size: int, hidden_size: int, num_output: int, num_layers: int = 1, dropout: float = 0.0, bidirectional: bool = False):
        """
        LSTM Model for sequence classification
        
        Args:
            input_size: Number of input features at each time step
            hidden_size: Number of hidden units in LSTM
            num_output: Number of output classes
            num_layers: Number of stacked LSTM layers
            dropout: Dropout probability (only applied if num_layers > 1)
            bidirectional: Whether to use bidirectional LSTM
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_output = num_output
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Classifier head
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        self.classifier = nn.Linear(lstm_output_size, num_output)
        
        self.dropout = nn.Dropout(dropout)
    
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, input_size)
        
        Returns:
            logits: Output tensor of shape (batch_size, num_output)
        """
        # LSTM forward pass
        # lstm_output shape: (batch_size, seq_length, hidden_size * num_directions)
        lstm_output, (hidden, cell) = self.lstm(x)
        
        # Take the last time step output for classification
        last_output = lstm_output[:, -1, :]  # (batch_size, hidden_size * num_directions)
        
        last_output = self.dropout(last_output)
        
        # Classification layer
        logits = self.classifier(last_output)
        
        return logits
