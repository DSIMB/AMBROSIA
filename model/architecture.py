activation_string_to_function = {
    'relu': F.relu,
    'tanh': F.tanh,
}

class CNN(nn.Module):
    def __init__(self, in_channels, hidden_layers=[512], dropout=0.1,
                 kernel_size=11, activation_function='relu'):
        super().__init__()
        if not isinstance(hidden_layers, list):
…
    def forward(self, x):
        """Forward pass through the convolutional layers."""
        for conv in self.conv_layers[:-1]:
            x = self.activation_function(self.dropout(conv(x)))
        x = self.conv_layers[-1](x)
        return x.squeeze()
