import torch
import torch.nn as nn


class MLP2(nn.Module):
    """
    Multi-layer perceptron with special architecture for pressure prediction
    """

    def __init__(self, network, activation=nn.Tanh()):
        """
        Initialize network

        Args:
            network: List defining network architecture [input_dim, hidden_dim1, ..., output_dim]
            activation: Activation function to use
        """
        super().__init__()
        self.network = network
        self.activation = activation

        # Shared layers for initial feature extraction
        self.shared_layers = nn.ModuleList()
        for i in range(len(network) - 3):
            layer = nn.Linear(network[i], network[i + 1])
            # xavier_initialization(layer)
            self.shared_layers.append(layer)

        # Split the network for velocity and pressure prediction
        mid_index = len(self.network) - 3
        mid_size = self.network[mid_index]

        # Velocity branch (2 outputs: u, v)
        self.velocity_branch = nn.Sequential(
            nn.Linear(mid_size, self.network[mid_index + 1]),
            self.activation,
            nn.Linear(self.network[mid_index + 1], 2),
        )

        # Pressure branch (1 output: p)
        # Use more capacity for pressure prediction
        pressure_mid_size = self.network[mid_index + 1] * 2
        self.pressure_branch = nn.Sequential(
            nn.Linear(mid_size, pressure_mid_size),
            self.activation,
            nn.Linear(pressure_mid_size, 1),
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Custom weight initialization for better convergence"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Xavier initialization with gain specific to tanh
                nn.init.xavier_normal_(m.weight, gain=5 / 3)
                nn.init.zeros_(m.bias)

    def forward(self, x, min_x=0.0, max_x=1.0):
        """
        Forward pass through the network

        Args:
            x: Input tensor

        Returns:
            Tensor with [u, v, p] predictions
        """
        # Process through shared layers
        x = 2 * (x - min_x) / (max_x - min_x) - 1
        for i, layer in enumerate(self.shared_layers):
            x = layer(x)
            x = self.activation(x)

        # Split into velocity and pressure branches
        velocity = self.velocity_branch(x)
        pressure = self.pressure_branch(x)

        # Concatenate outputs
        output = torch.cat([velocity, pressure], dim=1)

        return output
