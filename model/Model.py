import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Linear, ReLU, Conv1d, MultiheadAttention, BatchNorm1d

class SweetConv(torch.nn.Module):
    def __init__(self, in_channels=320, out_channels=2, hidden_layers=[128], 
                 dropout=0.5, dropout_conv=0.1, window_size=13, kernel_size=[3],
                 hidden_layers_fc=128):
        super().__init__()
        if len(hidden_layers) < 3:
            h1, h2, h3 = [hidden_layers[0]]*3
        else:
            h1, h2, h3 = hidden_layers
        if len(kernel_size) < 3:
            k1, k2, k3 = [kernel_size]*3
        else:
            k1, k2, k3 = kernel_size 
        self.conv1 = Conv1d(in_channels, h1, k1, padding='same')
        self.bn1 = BatchNorm1d(h1)
        self.conv2 = Conv1d(h1, h2, k2, padding='same')
        self.bn2 = BatchNorm1d(h2)
        self.conv3 = Conv1d(h2, h3, k3, padding='same') 
        self.bn3 = BatchNorm1d(h3)
        self.dropout_conv = nn.Dropout(p=dropout_conv)
        self.fc_size = h3*window_size
        self.fc1 = nn.Linear(self.fc_size, hidden_layers_fc)
        self.bn4 = BatchNorm1d(hidden_layers_fc)
        self.fc2 = nn.Linear(hidden_layers_fc, out_channels) 
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.dropout_conv(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout_conv(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout_conv(F.relu(self.bn3(self.conv3(x))))
        x = torch.flatten(x, 1) 
        x = self.dropout(F.relu(self.bn4(self.fc1(x))))
        x = F.relu(self.fc2(x))
        return x
