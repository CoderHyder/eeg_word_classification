import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGCNN(nn.Module):
    """
    EEGNet-style CNN for EEG classification.
    Input: (batch, n_channels=14, n_samples=128*2 approx)
    Output: class logits (batch, n_classes)
    """
    def __init__(self, n_channels=14, n_classes=5, input_samples=256):
        super(EEGCNN, self).__init__()
        
        # First temporal convolution
        self.conv1 = nn.Conv2d(1, 16, (1, 64), padding=(0, 32), bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        # Depthwise convolution (spatial filter across channels)
        self.depthwise = nn.Conv2d(16, 32, (n_channels, 1),
                                   groups=16, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.elu = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(0.5)

        # Separable convolution
        self.separable = nn.Conv2d(32, 32, (1, 16), padding=(0, 8), bias=False)
        self.bn3 = nn.BatchNorm2d(32)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(0.5)

        # Compute flatten size
        dummy_input = torch.zeros(1, 1, n_channels, input_samples)
        with torch.no_grad():
            out = self._forward_features(dummy_input)
            self.flat_features = out.shape[1]

        # Classifier
        self.fc = nn.Linear(self.flat_features, n_classes)

    def _forward_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.depthwise(x)
        x = self.bn2(x)
        x = self.elu(x)
        x = self.pool1(x)
        x = self.drop1(x)
        x = self.separable(x)
        x = self.bn3(x)
        x = self.elu(x)
        x = self.pool2(x)
        x = self.drop2(x)
        x = x.view(x.size(0), -1)
        return x

    def forward(self, x):
        # Expect input shape (batch, channels, samples)
        x = x.unsqueeze(1)  # (batch, 1, channels, samples)
        x = self._forward_features(x)
        x = self.fc(x)
        return x
