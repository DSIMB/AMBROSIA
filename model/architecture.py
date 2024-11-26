activation_string_to_function = {
    'relu': F.relu,
    'tanh': F.tanh,
}

class CNN(nn.Module):
    def __init__(self, in_channels, hidden_layers=[512], dropout=0.1, kernel_size=11, activation_function='relu'):
        super().__init__()
        if not isinstance(hidden_layers, list):
            hidden_layers = [hidden_layers]

# Initialize convolutional layers. We use a ModuleList to store them.
        hidden_layers = [in_channels] + hidden_layers + [1]
        self.conv_layers = nn.ModuleList()
        for n_layers, n_layers2 in zip(hidden_layers[:-1], hidden_layers[1:]):
            self.conv_layers.append(
                nn.Conv1d(n_layers, n_layers2, kernel_size=kernel_size, padding='same'))
        self.dropout = nn.Dropout(p=dropout)
        self.activation_function = activation_string_to_function.get(activation_function, 'relu')

    def forward(self, x):
        """Forward pass through the convolutional layers."""
        for conv in self.conv_layers[:-1]:
            x = self.activation_function(self.dropout(conv(x)))
        x = self.conv_layers[-1](x)
        return x.squeeze()
