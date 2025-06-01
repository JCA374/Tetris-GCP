# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    """
    DQN model adapted for smaller Tetris grid sizes.
    Uses smaller kernel sizes and strides to handle the 14x7 grid.
    """
    
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        
        self.input_shape = input_shape
        
        # Get input channels dynamically
        in_channels = input_shape[0]
        
        # Modified convolutional layers for smaller grid
        self.conv = nn.Sequential(
            # First layer: smaller 3x3 kernel with stride 1
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # Second layer
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # Third layer
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        
        # Calculate the size of the convolution output
        conv_out_size = self._get_conv_output(input_shape)
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
        
        # Initialize weights
        self._initialize_weights()

    def _get_conv_output(self, shape):
        """
        Calculate the output size of the convolutional layers
        to properly size the input to the fully connected layer.
        """
        with torch.no_grad():
            # Create a dummy tensor with the input shape
            dummy_input = torch.zeros(1, *shape)
            # Pass it through the convolutional layers
            conv_output = self.conv(dummy_input)
            # Return the flattened size
            return int(torch.prod(torch.tensor(conv_output.size()[1:])))
    
    def _initialize_weights(self):
        """Initialize weights using Kaiming normal initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """Forward pass through the network."""
        # Check if input channels match model's expectations
        expected_channels = self.conv[0].weight.shape[1]  # First conv layer's input channels
        actual_channels = x.shape[1]
        
        if expected_channels != actual_channels:
            print(f"\n*** CRITICAL: Input channel mismatch in forward pass! ***")
            print(f"Model expects {expected_channels} channels but got {actual_channels}")
            print("This usually happens when loading a model trained with a different preprocessing mode.")
            print("Please delete all checkpoint and model files and start fresh.")
            raise RuntimeError(f"Input channel mismatch: model expects {expected_channels} but got {actual_channels}. Delete checkpoints/models and restart training.")
        
        # Forward pass through convolutional layers
        conv_out = self.conv(x)
        # Flatten the output
        conv_out = conv_out.view(conv_out.size(0), -1)
        # Forward pass through fully connected layers
        return self.fc(conv_out)

class DuelDQN(nn.Module):
    """
    Dueling DQN model that separates value and advantage streams.
    Adapted for smaller Tetris grid sizes.
    """
    
    def __init__(self, input_shape, n_actions):
        super(DuelDQN, self).__init__()
        
        self.input_shape = input_shape
        
        # Get input channels dynamically
        in_channels = input_shape[0]
        
        # Modified convolutional layers for smaller grid
        self.conv = nn.Sequential(
            # First layer: smaller 3x3 kernel with stride 1
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # Second layer
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # Third layer
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        
        # Calculate the size of the convolution output
        conv_out_size = self._get_conv_output(input_shape)
        
        # Value stream - estimates V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        
        # Advantage stream - estimates A(s,a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
        
        # Initialize weights
        self._initialize_weights()

    def _get_conv_output(self, shape):
        """
        Calculate the output size of the convolutional layers
        to properly size the input to the fully connected layers.
        """
        with torch.no_grad():
            # Create a dummy tensor with the input shape
            dummy_input = torch.zeros(1, *shape)
            # Pass it through the convolutional layers
            conv_output = self.conv(dummy_input)
            # Return the flattened size
            return int(torch.prod(torch.tensor(conv_output.size()[1:])))
    
    def _initialize_weights(self):
        """Initialize weights using Kaiming normal initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """Forward pass combining value and advantage streams."""
        # Check if input channels match model's expectations
        expected_channels = self.conv[0].weight.shape[1]  # First conv layer's input channels
        actual_channels = x.shape[1]
        
        if expected_channels != actual_channels:
            print(f"\n*** CRITICAL: Input channel mismatch in forward pass! ***")
            print(f"Model expects {expected_channels} channels but got {actual_channels}")
            print("This usually happens when loading a model trained with a different preprocessing mode.")
            print("Please delete all checkpoint and model files and start fresh.")
            raise RuntimeError(f"Input channel mismatch: model expects {expected_channels} but got {actual_channels}. Delete checkpoints/models and restart training.")
        
        # Forward pass through convolutional layers
        conv_out = self.conv(x)
        # Flatten the output
        conv_out = conv_out.view(conv_out.size(0), -1)
        
        # Split into value and advantage streams
        value = self.value_stream(conv_out)
        advantage = self.advantage_stream(conv_out)
        
        # Combine streams: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        # This provides better numerical stability
        return value + (advantage - advantage.mean(dim=1, keepdim=True))