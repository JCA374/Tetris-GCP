import numpy as np
import gym
from gym import spaces
from simple_tetris_env import SimpleTetrisEnv, TETROMINO_SHAPES

class HighLevelTetrisEnv(gym.Wrapper):
    """
    A wrapper for SimpleTetrisEnv that provides high-level placement actions.
    Instead of controlling individual movements, the agent selects the final
    position and rotation for each piece.
    """
    
    def __init__(self, env=None, grid_width=7, grid_height=14):
        """Initialize the high-level Tetris environment."""
        # Create an environment if not provided
        if env is None:
            env = SimpleTetrisEnv(grid_width=grid_width, grid_height=grid_height)
        super().__init__(env)
        
        # Store original environment
        self.env = env
        self.grid_width = env.grid_width
        self.grid_height = env.grid_height
        
        # Maximum number of placements (rotation × position)
        # This is an estimate - we'll filter valid ones at runtime
        max_actions = 4 * self.grid_width  # 4 rotations × grid width
        
        # Replace the action space
        self.action_space = spaces.Discrete(max_actions)
        
        # Storage for valid actions
        self.valid_placements = []
        self.placement_to_actions = []
        
    def _find_landing_height(self, piece_shape, x):
        """Find the y position where the piece would land if dropped at column x."""
        # Start from the top
        for y in range(self.grid_height - len(piece_shape)):
            # Check if the piece would collide at this position
            for piece_y in range(len(piece_shape)):
                for piece_x in range(len(piece_shape[0])):
                    if piece_shape[piece_y][piece_x] == 0:
                        continue
                    
                    grid_y = y + piece_y + 1  # Check position below
                    grid_x = x + piece_x
                    
                    # Check if out of bounds or collision
                    if (grid_y >= self.grid_height or 
                        (0 <= grid_y < self.grid_height and 
                         0 <= grid_x < self.grid_width and 
                         self.env.grid[grid_y, grid_x] > 0)):
                        return y  # This is the landing height
        
        # If no collision found, place at the bottom
        return self.grid_height - len(piece_shape)
    
    def _generate_valid_placements(self):
        """Generate all valid final placements for the current piece."""
        self.valid_placements = []
        self.placement_to_actions = []
        
        # Get current piece details
        current_piece = self.env.current_piece
        if current_piece is None:
            return []
        
        # For each rotation
        for rotation in range(len(TETROMINO_SHAPES[current_piece])):
            piece_shape = TETROMINO_SHAPES[current_piece][rotation]
            piece_width = len(piece_shape[0])
            
            # For each possible x position
            for x in range(self.grid_width - piece_width + 1):
                # Find the y position where the piece would land
                y = self._find_landing_height(piece_shape, x)
                
                # Create placement (rotation, x, y)
                placement = (rotation, x, y)
                
                # Store the placement and the action sequence to achieve it
                self.valid_placements.append(placement)
                
                # Generate sequence of low-level actions to achieve this placement
                action_sequence = self._get_action_sequence(placement)
                self.placement_to_actions.append(action_sequence)
        
        return self.valid_placements
    
    def _get_action_sequence(self, placement):
        """Generate a sequence of low-level actions to achieve the given placement."""
        target_rotation, target_x, target_y = placement
        
        # Get current piece state
        current_rotation = self.env.current_rotation
        current_x = self.env.current_x
        current_y = self.env.current_y
        
        # Sequence of actions
        actions = []
        
        # 1. Rotate to the target rotation
        rotations_needed = (target_rotation - current_rotation) % len(TETROMINO_SHAPES[self.env.current_piece])
        actions.extend([2] * rotations_needed)  # Action 2 = rotate
        
        # 2. Move horizontally to target position
        if target_x < current_x:
            actions.extend([0] * (current_x - target_x))  # Action 0 = left
        else:
            actions.extend([1] * (target_x - current_x))  # Action 1 = right
        
        # 3. Drop the piece (Action 4 = hard drop)
        actions.append(4)
        
        return actions
    
    def reset(self):
        """Reset the environment and generate valid placements."""
        observation = self.env.reset()
        self._generate_valid_placements()
        return observation
    
    def step(self, action):
        """
        Execute a high-level placement action.
        
        Args:
            action: Index into the valid_placements list
            
        Returns:
            observation, reward, done, info
        """
        # Check if action is valid
        if not self.valid_placements:
            self._generate_valid_placements()
        
        # Cap action to valid range
        action = min(action, len(self.valid_placements) - 1)
        
        # Get the action sequence for this placement
        action_sequence = self.placement_to_actions[action]
        
        # Execute the sequence
        total_reward = 0
        info = {}
        last_info = {}
        
        for low_level_action in action_sequence:
            observation, reward, done, last_info = self.env.step(low_level_action)
            total_reward += reward
            
            if done:
                break
        
        # Update valid placements for the next piece
        if not done:
            self._generate_valid_placements()
        
        # Combine info dictionaries
        info = last_info
        info['high_level_action'] = action
        info['placement'] = self.valid_placements[action] if action < len(self.valid_placements) else None
        
        return observation, total_reward, done, info

def visualize_placements(env, top_k=5):
    """
    Create a visualization of the top-k possible placements.
    Useful for debugging and understanding what the agent sees.
    
    Args:
        env: A HighLevelTetrisEnv instance
        top_k: Number of placements to visualize
        
    Returns:
        List of grids showing the placements
    """
    if not hasattr(env, 'valid_placements'):
        return None
    
    # Generate placements if needed
    if not env.valid_placements:
        env._generate_valid_placements()
    
    # Limit to top-k (or fewer if not enough)
    num_placements = min(top_k, len(env.valid_placements))
    
    # Create visualizations
    viz_grids = []
    
    for i in range(num_placements):
        rotation, x, y = env.valid_placements[i]
        
        # Make a copy of the current grid
        grid_copy = env.env.grid.copy()
        
        # Get the piece shape
        piece_shape = TETROMINO_SHAPES[env.env.current_piece][rotation]
        
        # Place the piece on the grid copy
        for py in range(len(piece_shape)):
            for px in range(len(piece_shape[0])):
                if piece_shape[py][px] == 1:
                    grid_y = y + py
                    grid_x = x + px
                    if 0 <= grid_y < env.grid_height and 0 <= grid_x < env.grid_width:
                        grid_copy[grid_y, grid_x] = env.env.current_piece + 1
        
        viz_grids.append(grid_copy)
    
    return viz_grids