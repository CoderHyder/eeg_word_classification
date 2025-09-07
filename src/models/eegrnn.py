import torch
import torch.nn as nn

class EEGRNN(nn.Module):
    """LSTM/GRU-based model for EEG classification."""
    
    def __init__(self, n_channels=14, n_classes=5, hidden_size=128, n_layers=2, cell_type='lstm'):
        super().__init__()
        
        # RNN cell type
        rnn_cell = nn.LSTM if cell_type.lower() == 'lstm' else nn.GRU
        
        self.rnn = rnn_cell(
            input_size=n_channels,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=0.2 if n_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, 1),  # 2 for bidirectional
            nn.Tanh()
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, n_classes)
        )
        
    def forward(self, x):
        # Input shape: (batch, channels, samples)
        x = x.transpose(1, 2)  # -> (batch, samples, channels)
        
        # RNN + get all hidden states
        outputs, _ = self.rnn(x)  # outputs: (batch, samples, hidden_size*2)
        
        # Attention weights
        attn_weights = self.attention(outputs)  # (batch, samples, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)
        
        # Weighted sum of hidden states
        context = torch.bmm(outputs.transpose(1, 2), attn_weights)  # (batch, hidden_size*2, 1)
        context = context.squeeze(-1)  # (batch, hidden_size*2)
        
        # Classification
        output = self.classifier(context)
        
        return output
