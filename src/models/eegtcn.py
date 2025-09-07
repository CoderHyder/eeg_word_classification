import torch
import torch.nn as nn

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding):
        super().__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                              stride=stride, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                              stride=stride, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu2(out)
        out = self.dropout2(out)
        
        return out

class EEGTCN(nn.Module):
    def __init__(self, n_channels=14, n_classes=5, channel_dims=[32, 64, 128]):
        super().__init__()
        
        # Initial channel projection
        self.conv1 = nn.Conv1d(n_channels, channel_dims[0], 1)
        
        # Temporal blocks with increasing dilation
        self.temporal_blocks = nn.ModuleList([
            TemporalBlock(
                channel_dims[i], 
                channel_dims[i+1] if i < len(channel_dims)-1 else channel_dims[-1],
                kernel_size=3,
                stride=1,
                dilation=2**i,
                padding=2**i
            ) for i in range(len(channel_dims))
        ])
        
        # Global average pooling and classification
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(channel_dims[-1], n_classes)
        
    def forward(self, x):
        # Input shape: (batch, channels, samples)
        x = self.conv1(x)
        
        for block in self.temporal_blocks:
            x = block(x)
            
        x = self.gap(x).squeeze(-1)
        x = self.classifier(x)
        
        return x
