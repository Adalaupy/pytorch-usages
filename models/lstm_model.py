"""
Long Short-Term Memory (LSTM) Network:

==================================================================================
Common Usage
==================================================================================

- Sequence classification
- Time series prediction
- Sequential data analysis


==================================================================================
Parameter Explanation:
==================================================================================

1. Dropout : Connect between each layer
- num_layers=1: [LSTM] → No dropout (no layers to connect)
- num_layers=2: [LSTM] → [dropout] → [LSTM] → dropout applied between them
- num_layers=3: [LSTM] → [dropout] → [LSTM] → [dropout] → [LSTM]


2. Bidirectional Sample cases
- Unidirectional (only left to right): 
    => Seq2Seq Translation (from left to right)
    => Stock Price Prediction (only past data provided)
- Bidirectional (both):
    => Named Entity Recognition : identify people, organization, locations etc
    => Sentiment Analysis : full context to identify -ve/+ve
    => Speech recognition : better for offline

    
3. batch_first : Control input/output shape convention
- Target for : "x = torch.randn(20, 32, 10)"
- True  : [batch_size, seq_len, input_size]
- False : [seq_len, batch_size, input_size]


4. Classifier : map hidden features to classess


==================================================================================
Others:
==================================================================================

1. For Machine Translation (Seq2Seq), there are encoder and decoder
- Encoder : read source and build meaning
- Decoder : Use the meaning to generate sentense
- Example of usages:
    => Only Encoder : understand and scoring, Token labeling, Classification of spam, sentiment, intent
    => Only Decoder : rewrite email, decoder also can understand meaning
    => Encoder + Decoder : translate, summarize, use when you need strong source to target mapping


    

"""


import torch.nn as nn
import torch.nn.functional as F


class LSTM_Model(nn.Module):
    
    def __init__(self, input_size: int, hidden_size: int, num_output: int, num_layers: int = 1, dropout: float = 0.0, bidirectional: bool = False):

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
