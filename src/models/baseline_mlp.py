import torch
import torch.nn as nn

class BaselineMLP(nn.Module):
    """Simple fully connected baseline model."""
    
    def __init__(self, n_channels=14, n_samples=256, n_classes=5, hidden_dims=[256, 128]):
        super().__init__()
        
        layers = []
        input_dim = n_channels * n_samples
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.5)
            ])
            input_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], n_classes))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        # Input shape: (batch, channels, samples)
        x = x.view(x.size(0), -1)  # Flatten: (batch, channels*samples)
        return self.model(x)
