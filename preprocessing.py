"""
Standardized preprocessing module for Tetris DQN project.

This module consolidates all preprocessing functions and provides:
- Single preprocess_state function
- GPU-optimized batch preprocessing  
- Consistent channel ordering
- Clear documentation of output format
"""
import numpy as np
import torch
from typing import Union, List, Optional, Dict, Any
import warnings

# Import Tetris shapes for piece rendering
try:
    from simple_tetris_env import TETROMINO_SHAPES
except ImportError:
    warnings.warn("Could not import TETROMINO_SHAPES, piece rendering will be disabled")
    TETROMINO_SHAPES = []


def preprocess_state(
    state: Union[Dict[str, Any], np.ndarray],
    binary: bool = False,
    include_piece_info: bool = True,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Process observation into tensor format for DQN with improved state representation.
    
    Args:
        state: Observation from environment (dict with grid, piece info OR numpy array)
        binary: Whether to use binary state representation (occupied vs empty)
        include_piece_info: Whether to include additional channels for piece information
        device: Device to place the tensor on ('cpu' or 'cuda')
        dtype: PyTorch data type for the output tensor
        
    Returns:
        Processed tensor in format [C, H, W] where:
        - If include_piece_info=False: C=1 (grid only)
        - If include_piece_info=True: C=4 (grid + piece_position + next_piece + rotation)
        
    Channel descriptions:
        - Channel 0: Grid state (occupied cells)
        - Channel 1: Current piece position (if include_piece_info=True)
        - Channel 2: Next piece preview (if include_piece_info=True)  
        - Channel 3: Rotation encoding (if include_piece_info=True)
    """
    # Handle simple grid-only preprocessing
    if not include_piece_info or not isinstance(state, dict):
        return _preprocess_grid_only(state, binary, device, dtype)
    
    # Enhanced preprocessing with piece information
    if not TETROMINO_SHAPES:
        warnings.warn("TETROMINO_SHAPES not available, falling back to grid-only preprocessing")
        return _preprocess_grid_only(state, binary, device, dtype)
    
    return _preprocess_enhanced(state, binary, device, dtype)


def _preprocess_grid_only(
    state: Union[Dict[str, Any], np.ndarray], 
    binary: bool, 
    device: str, 
    dtype: torch.dtype
) -> torch.Tensor:
    """Process grid data only without piece information."""
    # Extract grid data
    if isinstance(state, dict) and 'grid' in state:
        grid = state['grid']
    else:
        grid = state
    
    # Ensure it's a numpy array
    if not isinstance(grid, np.ndarray):
        grid = np.array(grid)
    
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
    
    # Convert to PyTorch tensor
    tensor = torch.from_numpy(grid).to(dtype=dtype, device=device)
    
    return tensor


def _preprocess_enhanced(
    state: Dict[str, Any], 
    binary: bool, 
    device: str, 
    dtype: torch.dtype
) -> torch.Tensor:
    """Process state with enhanced piece information."""
    # Extract state components
    grid = state['grid']
    current_piece = state.get('current_piece')
    piece_x = state.get('piece_x', 0)
    piece_y = state.get('piece_y', 0)
    piece_rotation = state.get('piece_rotation', 0)
    next_piece = state.get('next_piece')
    
    # Ensure grid is numpy array
    if not isinstance(grid, np.ndarray):
        grid = np.array(grid)
    
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
    
    # Channel 0: Grid state
    grid_channel = np.transpose(grid, (2, 0, 1)).astype(np.float32)
    
    # Channel 1: Current piece position
    piece_pos_channel = np.zeros((1, height, width), dtype=np.float32)
    if current_piece is not None and piece_rotation is not None:
        _render_piece_on_channel(
            piece_pos_channel, current_piece, piece_rotation, 
            piece_x, piece_y, height, width, intensity=1.0
        )
    
    # Channel 2: Next piece preview
    next_piece_channel = np.zeros((1, height, width), dtype=np.float32)
    if next_piece is not None:
        # Position preview at top center of grid
        preview_x = width // 2 - 2  # Approximate center
        preview_y = 1  # Near the top
        _render_piece_on_channel(
            next_piece_channel, next_piece, 0,  # Use rotation 0 for preview
            preview_x, preview_y, height, width, intensity=0.5
        )
    
    # Channel 3: Rotation encoding
    rotation_channel = np.zeros((1, height, width), dtype=np.float32)
    if piece_rotation is not None:
        # Normalize rotation to 0-1 range (max rotation is 3)
        rotation_channel[0, 0, 0] = piece_rotation / 3.0
    
    # Combine all channels
    combined = np.vstack([
        grid_channel,
        piece_pos_channel,
        next_piece_channel,
        rotation_channel
    ])
    
    # Convert to PyTorch tensor
    tensor = torch.from_numpy(combined).to(dtype=dtype, device=device)
    
    return tensor


def _render_piece_on_channel(
    channel: np.ndarray,
    piece_type: int,
    rotation: int,
    pos_x: int,
    pos_y: int,
    height: int,
    width: int,
    intensity: float = 1.0
) -> None:
    """Render a tetromino piece onto a channel at specified position."""
    if not (0 <= piece_type < len(TETROMINO_SHAPES)):
        return
    
    if not (0 <= rotation < len(TETROMINO_SHAPES[piece_type])):
        return
    
    piece_shape = TETROMINO_SHAPES[piece_type][rotation]
    
    # Map the piece onto the channel at the specified position
    for y in range(len(piece_shape)):
        for x in range(len(piece_shape[y])):
            if piece_shape[y][x] == 1:
                grid_y = pos_y + y
                grid_x = pos_x + x
                if 0 <= grid_y < height and 0 <= grid_x < width:
                    channel[0, grid_y, grid_x] = intensity


class BatchPreprocessor:
    """
    Efficiently preprocesses batches of states with GPU acceleration.
    Provides vectorized preprocessing for improved performance.
    """
    
    def __init__(
        self,
        device: str = "cuda",
        binary: bool = False,
        include_piece_info: bool = True,
        dtype: torch.dtype = torch.float32,
        pin_memory: bool = True,
        non_blocking: bool = True
    ):
        """
        Initialize the batch preprocessor.
        
        Args:
            device: Device to place tensors on
            binary: Whether to use binary state representation
            include_piece_info: Whether to include additional channels for piece info
            dtype: PyTorch data type for output tensors
            pin_memory: Whether to use pinned memory for faster CPU-GPU transfers
            non_blocking: Whether to use non-blocking transfers
        """
        self.device = device
        self.binary = binary
        self.include_piece_info = include_piece_info
        self.dtype = dtype
        self.pin_memory = pin_memory
        self.non_blocking = non_blocking
        
    def preprocess_batch(self, states: List[Any]) -> Optional[torch.Tensor]:
        """
        Preprocess a batch of states efficiently.
        
        Args:
            states: List of observations from environments
            
        Returns:
            Batch tensor of shape [B, C, H, W] on the device, or None if no valid states
        """
        # Filter out None states (terminal states)
        valid_states = [state for state in states if state is not None]
        
        if not valid_states:
            return None
        
        # Process each state
        processed_states = []
        for state in valid_states:
            if isinstance(state, torch.Tensor):
                # Already a tensor, ensure it's on the right device and dtype
                tensor = state.to(device=self.device, dtype=self.dtype, non_blocking=self.non_blocking)
            else:
                # Preprocess the state
                tensor = preprocess_state(
                    state, 
                    binary=self.binary,
                    include_piece_info=self.include_piece_info,
                    device='cpu',  # Process on CPU first for efficiency
                    dtype=self.dtype
                )
                
                # Move to target device
                if self.pin_memory and tensor.device.type == 'cpu':
                    tensor = tensor.pin_memory()
                tensor = tensor.to(device=self.device, non_blocking=self.non_blocking)
            
            processed_states.append(tensor)
        
        # Stack into batch tensor
        return torch.stack(processed_states)
    
    def preprocess_single(self, state: Any) -> torch.Tensor:
        """
        Preprocess a single state with the batch preprocessor settings.
        
        Args:
            state: Single observation from environment
            
        Returns:
            Processed tensor on the target device
        """
        if isinstance(state, torch.Tensor):
            return state.to(device=self.device, dtype=self.dtype, non_blocking=self.non_blocking)
        
        tensor = preprocess_state(
            state,
            binary=self.binary,
            include_piece_info=self.include_piece_info,
            device='cpu',
            dtype=self.dtype
        )
        
        if self.pin_memory and tensor.device.type == 'cpu':
            tensor = tensor.pin_memory()
        
        return tensor.to(device=self.device, non_blocking=self.non_blocking)


def get_state_shape(include_piece_info: bool = True, grid_height: int = 14, grid_width: int = 7) -> tuple:
    """
    Get the expected shape of preprocessed states.
    
    Args:
        include_piece_info: Whether piece information channels are included
        grid_height: Height of the Tetris grid
        grid_width: Width of the Tetris grid
        
    Returns:
        Tuple representing the shape (C, H, W)
    """
    channels = 4 if include_piece_info else 1
    return (channels, grid_height, grid_width)


def validate_state_format(tensor: torch.Tensor, include_piece_info: bool = True) -> bool:
    """
    Validate that a preprocessed state tensor has the expected format.
    
    Args:
        tensor: Preprocessed state tensor
        include_piece_info: Whether piece information channels are expected
        
    Returns:
        True if the tensor has the expected format, False otherwise
    """
    if not isinstance(tensor, torch.Tensor):
        return False
    
    if len(tensor.shape) != 3:  # Should be [C, H, W]
        return False
    
    expected_channels = 4 if include_piece_info else 1
    if tensor.shape[0] != expected_channels:
        return False
    
    return True


# Factory function for easy preprocessor creation
def create_preprocessor(
    device: str = "cuda",
    preprocessing_type: str = "enhanced",
    **kwargs
) -> Union[BatchPreprocessor, callable]:
    """
    Factory function to create preprocessors with common configurations.
    
    Args:
        device: Device to use for processing
        preprocessing_type: Type of preprocessing ('basic', 'enhanced', 'batch')
        **kwargs: Additional arguments for the preprocessor
        
    Returns:
        Configured preprocessor
    """
    if preprocessing_type == "basic":
        def basic_preprocessor(state):
            return preprocess_state(state, include_piece_info=False, device=device, **kwargs)
        return basic_preprocessor
    
    elif preprocessing_type == "enhanced":
        def enhanced_preprocessor(state):
            return preprocess_state(state, include_piece_info=True, device=device, **kwargs)
        return enhanced_preprocessor
    
    elif preprocessing_type == "batch":
        return BatchPreprocessor(device=device, include_piece_info=True, **kwargs)
    
    elif preprocessing_type == "batch_basic":
        return BatchPreprocessor(device=device, include_piece_info=False, **kwargs)
    
    else:
        raise ValueError(f"Unknown preprocessing_type: {preprocessing_type}. "
                        f"Choose from 'basic', 'enhanced', 'batch', 'batch_basic'")