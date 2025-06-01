import numpy as np
import torch
from simple_tetris_env import TETROMINO_SHAPES

def preprocess_state_gpu(state, binary=False, include_piece_info=True, device="cuda"):
    """
    Process observation into tensor format for DQN with GPU optimization.
    Returns tensors directly on the GPU to eliminate CPU-GPU transfer overhead.
    
    Args:
        state: Observation from environment
        binary: Whether to use binary state representation
        include_piece_info: Whether to include additional channels for piece info
        device: Device to place tensors on
        
    Returns:
        Processed tensor in format [C, H, W] with optional additional channels,
        placed directly on the GPU
    """
    # If include_piece_info is False or the state is not a dictionary, use the original method
    if not include_piece_info or not isinstance(state, dict):
        # Handle standard grid data
        if isinstance(state, dict) and 'grid' in state:
            grid = state['grid']
        else:
            grid = state

        # Add channel dimension if it's 2D
        if len(grid.shape) == 2:
            grid = grid.reshape(grid.shape[0], grid.shape[1], 1)

        # Apply binary conversion if requested
        if binary:
            grid = (grid > 0).astype(np.float32)
        else:
            grid = grid.astype(np.float32)

        # Transpose to [C, H, W] format expected by CNN
        grid = np.transpose(grid, (2, 0, 1))
        
        # Convert to PyTorch tensor on GPU
        return torch.from_numpy(grid).to(device)

    # Enhanced preprocessing with piece information
    # Extract the grid and piece information
    grid = state['grid']
    current_piece = state.get('current_piece')
    piece_x = state.get('piece_x', 0)
    piece_y = state.get('piece_y', 0)
    piece_rotation = state.get('piece_rotation', 0)
    next_piece = state.get('next_piece')

    # Add channel dimension if it's 2D
    if len(grid.shape) == 2:
        grid = grid.reshape(grid.shape[0], grid.shape[1], 1)

    # Apply binary conversion if requested
    if binary:
        grid = (grid > 0).astype(np.float32)
    else:
        grid = grid.astype(np.float32)

    # Get grid dimensions
    height, width = grid.shape[0], grid.shape[1]

    # Channel 1: Grid state (possibly multi-channel already)
    grid_channel = np.transpose(grid, (2, 0, 1)).astype(np.float32)

    # Channel 2: Current piece position
    piece_pos_channel = np.zeros((1, height, width), dtype=np.float32)

    # Render the actual current piece shape at its current position
    if current_piece is not None and piece_rotation is not None:
        # Get the current piece shape based on type and rotation
        if 0 <= current_piece < len(TETROMINO_SHAPES) and 0 <= piece_rotation < len(TETROMINO_SHAPES[current_piece]):
            piece_shape = TETROMINO_SHAPES[current_piece][piece_rotation]

            # Map the piece onto the grid at its current position
            for y in range(len(piece_shape)):
                for x in range(len(piece_shape[y])):
                    if piece_shape[y][x] == 1:
                        grid_y = piece_y + y
                        grid_x = piece_x + x
                        if 0 <= grid_y < height and 0 <= grid_x < width:
                            piece_pos_channel[0, grid_y, grid_x] = 1.0

    # Channel 3: Next piece preview
    next_piece_channel = np.zeros((1, height, width), dtype=np.float32)

    if next_piece is not None and 0 <= next_piece < len(TETROMINO_SHAPES):
        # Get the shape of the next piece (first rotation)
        next_shape = TETROMINO_SHAPES[next_piece][0]

        # Position it at the top middle of the grid for preview
        preview_x = width // 2 - len(next_shape[0]) // 2
        preview_y = 1  # Near the top

        # Map the next piece shape onto the grid
        for y in range(len(next_shape)):
            for x in range(len(next_shape[y])):
                if next_shape[y][x] == 1:
                    grid_y = preview_y + y
                    grid_x = preview_x + x
                    if 0 <= grid_y < height and 0 <= grid_x < width:
                        # Use half intensity to differentiate
                        next_piece_channel[0, grid_y, grid_x] = 0.5

    # Channel 4: Rotation encoding (simplified to a single value)
    rotation_channel = np.zeros((1, height, width), dtype=np.float32)
    if piece_rotation is not None:
        # Normalize to 0-1 range (max rotation is 3)
        rotation_channel[0, 0, 0] = piece_rotation / 3.0

    # Combine all channels
    combined = np.vstack([
        grid_channel,
        piece_pos_channel,
        next_piece_channel,
        rotation_channel
    ])

    # Convert to PyTorch tensor directly on GPU
    return torch.from_numpy(combined).to(device)

class BatchPreprocessor:
    """
    Efficiently preprocesses batches of states with GPU acceleration.
    Can be used to preprocess a batch of states in parallel.
    """
    
    def __init__(self, device="cuda", binary=False, include_piece_info=True, 
                 pin_memory=True, non_blocking=True):
        """
        Initialize the batch preprocessor.
        
        Args:
            device: Device to place tensors on
            binary: Whether to use binary state representation
            include_piece_info: Whether to include additional channels for piece info
            pin_memory: Whether to use pinned memory for faster CPU-GPU transfers
            non_blocking: Whether to use non-blocking transfers
        """
        self.device = device
        self.binary = binary
        self.include_piece_info = include_piece_info
        self.pin_memory = pin_memory
        self.non_blocking = non_blocking
        
    def preprocess_batch(self, states):
        """
        Preprocess a batch of states efficiently.
        
        Args:
            states: List of observations from environments
            
        Returns:
            Batch tensor of shape [B, C, H, W] on the device
        """
        # Handle None elements in the batch (for terminal states)
        processed_states = []
        
        for state in states:
            if state is None:
                continue
                
            if isinstance(state, torch.Tensor):
                # If already a tensor, just ensure it's on the right device
                if state.device.type != self.device:
                    processed_states.append(state.to(self.device, non_blocking=self.non_blocking))
                else:
                    processed_states.append(state)
            else:
                # Process the state
                tensor = self._preprocess_single_state(state)
                processed_states.append(tensor)
                
        if not processed_states:
            # No valid states in the batch
            return None
            
        # Stack tensors into a batch
        return torch.stack(processed_states)
    
    def _preprocess_single_state(self, state):
        """
        Process a single state into a tensor.
        
        Args:
            state: Observation from the environment
            
        Returns:
            Processed tensor
        """
        # If include_piece_info is False or the state is not a dictionary, use the original method
        if not self.include_piece_info or not isinstance(state, dict):
            # Handle standard grid data
            if isinstance(state, dict) and 'grid' in state:
                grid = state['grid']
            else:
                grid = state

            # Add channel dimension if it's 2D
            if len(grid.shape) == 2:
                grid = grid.reshape(grid.shape[0], grid.shape[1], 1)

            # Apply binary conversion if requested
            if self.binary:
                grid = (grid > 0).astype(np.float32)
            else:
                grid = grid.astype(np.float32)

            # Transpose to [C, H, W] format expected by CNN
            grid = np.transpose(grid, (2, 0, 1))
            
            # Convert to PyTorch tensor on GPU with optional pinning
            tensor = torch.from_numpy(grid)
            if self.pin_memory and tensor.device.type == 'cpu':
                tensor = tensor.pin_memory()
            return tensor.to(self.device, non_blocking=self.non_blocking)

        # Enhanced preprocessing with piece information
        # Extract the grid and piece information
        grid = state['grid']
        current_piece = state.get('current_piece')
        piece_x = state.get('piece_x', 0)
        piece_y = state.get('piece_y', 0)
        piece_rotation = state.get('piece_rotation', 0)
        next_piece = state.get('next_piece')

        # Add channel dimension if it's 2D
        if len(grid.shape) == 2:
            grid = grid.reshape(grid.shape[0], grid.shape[1], 1)

        # Apply binary conversion if requested
        if self.binary:
            grid = (grid > 0).astype(np.float32)
        else:
            grid = grid.astype(np.float32)

        # Get grid dimensions
        height, width = grid.shape[0], grid.shape[1]

        # Channel 1: Grid state (possibly multi-channel already)
        grid_channel = np.transpose(grid, (2, 0, 1)).astype(np.float32)

        # Channel 2: Current piece position
        piece_pos_channel = np.zeros((1, height, width), dtype=np.float32)

        # Render the actual current piece shape at its current position
        if current_piece is not None and piece_rotation is not None:
            # Get the current piece shape based on type and rotation
            if 0 <= current_piece < len(TETROMINO_SHAPES) and 0 <= piece_rotation < len(TETROMINO_SHAPES[current_piece]):
                piece_shape = TETROMINO_SHAPES[current_piece][piece_rotation]

                # Map the piece onto the grid at its current position
                for y in range(len(piece_shape)):
                    for x in range(len(piece_shape[y])):
                        if piece_shape[y][x] == 1:
                            grid_y = piece_y + y
                            grid_x = piece_x + x
                            if 0 <= grid_y < height and 0 <= grid_x < width:
                                piece_pos_channel[0, grid_y, grid_x] = 1.0

        # Channel 3: Next piece preview
        next_piece_channel = np.zeros((1, height, width), dtype=np.float32)

        if next_piece is not None and 0 <= next_piece < len(TETROMINO_SHAPES):
            # Get the shape of the next piece (first rotation)
            next_shape = TETROMINO_SHAPES[next_piece][0]

            # Position it at the top middle of the grid for preview
            preview_x = width // 2 - len(next_shape[0]) // 2
            preview_y = 1  # Near the top

            # Map the next piece shape onto the grid
            for y in range(len(next_shape)):
                for x in range(len(next_shape[y])):
                    if next_shape[y][x] == 1:
                        grid_y = preview_y + y
                        grid_x = preview_x + x
                        if 0 <= grid_y < height and 0 <= grid_x < width:
                            # Use half intensity to differentiate
                            next_piece_channel[0, grid_y, grid_x] = 0.5

        # Channel 4: Rotation encoding (simplified to a single value)
        rotation_channel = np.zeros((1, height, width), dtype=np.float32)
        if piece_rotation is not None:
            # Normalize to 0-1 range (max rotation is 3)
            rotation_channel[0, 0, 0] = piece_rotation / 3.0

        # Combine all channels
        combined = np.vstack([
            grid_channel,
            piece_pos_channel,
            next_piece_channel,
            rotation_channel
        ])

        # Convert to PyTorch tensor with optional pinning
        tensor = torch.from_numpy(combined)
        if self.pin_memory and tensor.device.type == 'cpu':
            tensor = tensor.pin_memory()
        return tensor.to(self.device, non_blocking=self.non_blocking)