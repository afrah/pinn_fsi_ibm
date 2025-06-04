import torch
import torch.nn as nn
from src.nn.bspline import KANLinear  # Import your KAN implementation


class KAN2(nn.Module):
    """
    Branched Kolmogorov-Arnold Network with separate paths for velocity and pressure
    """

    def __init__(
        self,
        network,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=nn.SiLU,
        grid_eps=0.02,
        grid_range=[-0.8, 0.8],
    ):
        """
        Initialize network with branched architecture

        Args:
            network: List defining network architecture [input_dim, hidden_dim1, ..., output_dim]
            grid_size: Number of grid points for KAN
            spline_order: Order of B-spline basis functions
            scale_noise: Scale of initialization noise
            scale_base: Scale of base activation
            scale_spline: Scale of spline
            base_activation: Base activation function
            grid_eps: Grid epsilon
            grid_range: Range of grid
        """
        super().__init__()
        self.network = network

        # Determine the split point for shared vs. specialized layers
        mid_index = len(network) - 3

        # Create shared layers using KANLinear
        self.shared_layers = nn.ModuleList()
        for i in range(mid_index):
            self.shared_layers.append(
                KANLinear(
                    network[i],
                    network[i + 1],
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

        # Dimensions for the branches
        mid_size = network[mid_index]
        hidden_size = network[mid_index + 1]

        # Velocity branch (2 outputs: u, v)
        self.velocity_branch = nn.Sequential(
            KANLinear(
                mid_size,
                hidden_size,
                grid_size=grid_size,
                spline_order=spline_order,
                scale_noise=scale_noise,
                scale_base=scale_base,
                scale_spline=scale_spline,
                base_activation=base_activation,
                grid_eps=grid_eps,
                grid_range=grid_range,
            ),
            KANLinear(
                hidden_size,
                2,  # Output size for u, v
                grid_size=grid_size,
                spline_order=spline_order,
                scale_noise=scale_noise,
                scale_base=scale_base,
                scale_spline=scale_spline,
                base_activation=base_activation,
                grid_eps=grid_eps,
                grid_range=grid_range,
            ),
        )

        # Pressure branch with increased capacity
        pressure_hidden_size = hidden_size * 2
        self.pressure_branch = nn.Sequential(
            KANLinear(
                mid_size,
                pressure_hidden_size,
                grid_size=grid_size,
                spline_order=spline_order,
                scale_noise=scale_noise,
                scale_base=scale_base,
                scale_spline=scale_spline,
                base_activation=base_activation,
                grid_eps=grid_eps,
                grid_range=grid_range,
            ),
            KANLinear(
                pressure_hidden_size,
                pressure_hidden_size,
                grid_size=grid_size,
                spline_order=spline_order,
                scale_noise=scale_noise,
                scale_base=scale_base,
                scale_spline=scale_spline,
                base_activation=base_activation,
                grid_eps=grid_eps,
                grid_range=grid_range,
            ),
            KANLinear(
                pressure_hidden_size,
                1,  # Output size for pressure
                grid_size=grid_size,
                spline_order=spline_order,
                scale_noise=scale_noise,
                scale_base=scale_base,
                scale_spline=scale_spline,
                base_activation=base_activation,
                grid_eps=grid_eps,
                grid_range=grid_range,
            ),
        )

    def forward(self, x, update_grid=False):
        """
        Forward pass through the network

        Args:
            x: Input tensor
            update_grid: Whether to update the grid

        Returns:
            Tensor with [u, v, p] predictions
        """
        # Process through shared layers
        for layer in self.shared_layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)

        # Process through velocity branch
        velocity_features = x
        for i, layer in enumerate(self.velocity_branch):
            if update_grid and isinstance(layer, KANLinear):
                layer.update_grid(velocity_features)
            velocity_features = layer(velocity_features)

        # Process through pressure branch
        pressure_features = x
        for i, layer in enumerate(self.pressure_branch):
            if update_grid and isinstance(layer, KANLinear):
                layer.update_grid(pressure_features)
            pressure_features = layer(pressure_features)

        # Concatenate outputs
        output = torch.cat([velocity_features, pressure_features], dim=1)

        return output

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss for all KANLinear layers
        """
        loss = 0.0

        # Shared layers
        for layer in self.shared_layers:
            if isinstance(layer, KANLinear):
                loss += layer.regularization_loss(
                    regularize_activation, regularize_entropy
                )

        # Velocity branch
        for layer in self.velocity_branch:
            if isinstance(layer, KANLinear):
                loss += layer.regularization_loss(
                    regularize_activation, regularize_entropy
                )

        # Pressure branch
        for layer in self.pressure_branch:
            if isinstance(layer, KANLinear):
                loss += layer.regularization_loss(
                    regularize_activation, regularize_entropy
                )

        return loss

    def update_grid(self, x):
        """
        Update the grid for all KANLinear layers
        """
        # Shared layers
        for i, layer in enumerate(self.shared_layers):
            if i == 0:
                layer.update_grid(x)
            else:
                # Get output from previous layer
                for j in range(i):
                    x = self.shared_layers[j](x)
                layer.update_grid(x)

        # Get features after shared layers
        shared_features = x

        # Velocity branch
        velocity_features = shared_features
        for i, layer in enumerate(self.velocity_branch):
            if isinstance(layer, KANLinear):
                if i == 0:
                    layer.update_grid(velocity_features)
                else:
                    # Get output from previous layers in branch
                    for j in range(i):
                        velocity_features = self.velocity_branch[j](velocity_features)
                    layer.update_grid(velocity_features)

        # Pressure branch
        pressure_features = shared_features
        for i, layer in enumerate(self.pressure_branch):
            if isinstance(layer, KANLinear):
                if i == 0:
                    layer.update_grid(pressure_features)
                else:
                    # Get output from previous layers in branch
                    for j in range(i):
                        pressure_features = self.pressure_branch[j](pressure_features)
                    layer.update_grid(pressure_features)
