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


from typing import Optional

import torch.nn as nn


class LSTM_Model(nn.Module):
    
    def __init__(
        self,
        input_size: Optional[int],
        hidden_size: int,
        num_output: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
        vocab_size: Optional[int] = None,
        embedding_dim: Optional[int] = None,
        padding_idx: int = 0,
    ):

        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_output = num_output
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

        # General-purpose input adapter:
        # - numeric mode: feed float features directly
        # - text mode: map token ids to dense vectors with embedding
        self.use_embedding = vocab_size is not None and embedding_dim is not None

        if self.use_embedding:
            self.embedding = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=embedding_dim,
                padding_idx=padding_idx,
            )
            lstm_input_size = embedding_dim
        else:
            if input_size is None:
                raise ValueError("input_size must be provided in numeric mode.")
            self.embedding = None
            lstm_input_size = input_size
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
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
    
    
    def forward(self, x, mask=None):
        """
        Forward pass
        
        Args:
            x:
                - numeric mode: (batch_size, seq_length, input_size)
                - text mode: (batch_size, seq_length) token ids
            mask: (batch_size, seq_length) boolean mask where True means valid token, False means padding
                  If None, assumes all positions are valid (for backward compatibility)
        
        Returns:
            logits: Output tensor of shape (batch_size, num_output)
        """
        # ------------------------------------------------------------------------
        # Forward flow (text mode): token ids -> embedding -> LSTM -> masked mean
        # pooling over non-padding tokens -> dropout -> classifier.
        # If mask is not provided, it is auto-built from padding_idx.
        # Forward flow (numeric mode): features -> LSTM -> last timestep output ->
        # dropout -> classifier.
        # ------------------------------------------------------------------------
        if self.use_embedding:
            x_ids = x.long()
            x = self.embedding(x_ids)

            if mask is None:
                mask = (x_ids != self.padding_idx)

        lstm_output, (hidden, cell) = self.lstm(x)

        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)
            masked_output = lstm_output * mask_expanded
            sum_output = masked_output.sum(dim=1)
            seq_lengths = mask.sum(dim=1, keepdim=True).clamp(min=1)
            last_output = sum_output / seq_lengths
        else:
            last_output = lstm_output[:, -1, :]
        
        last_output = self.dropout(last_output)
        
        logits = self.classifier(last_output)
        
        return logits
