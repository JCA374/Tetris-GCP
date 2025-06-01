"""
Simple Tetris environment implementation that follows the Gym interface.
This implementation is much easier to set up than gym-tetris.
"""
import gym
import numpy as np
import pygame
import random
import time
from gym import spaces
from config import CONFIG  # Import CONFIG at the top of the file

# Define the shapes of the tetrominos
SHAPES = [
    [[1, 1, 1, 1]],           # I
    [[1, 1], [1, 1]],         # O
    [[1, 1, 1], [0, 1, 0]],   # T
    [[1, 1, 1], [1, 0, 0]],   # J
    [[1, 1, 1], [0, 0, 1]],   # L
    [[0, 1, 1], [1, 1, 0]],   # S
    [[1, 1, 0], [0, 1, 1]]    # Z
]

# Define all possible tetromino shapes and their rotations
TETROMINO_SHAPES = [
    # I tetromino - has 2 rotation states
    [
        [[1, 1, 1, 1]],
        [[1], [1], [1], [1]]
    ],
    # O tetromino - has 1 rotation state (doesn't change when rotated)
    [
        [[1, 1], [1, 1]]
    ],
    # T tetromino - has 4 rotation states
    [
        [[1, 1, 1], [0, 1, 0]],
        [[0, 1], [1, 1], [0, 1]],
        [[0, 1, 0], [1, 1, 1]],
        [[1, 0], [1, 1], [1, 0]]
    ],
    # J tetromino - has 4 rotation states
    [
        [[1, 1, 1], [0, 0, 1]],
        [[0, 1], [0, 1], [1, 1]],
        [[1, 0, 0], [1, 1, 1]],
        [[1, 1], [1, 0], [1, 0]]
    ],
    # L tetromino - has 4 rotation states
    [
        [[1, 1, 1], [1, 0, 0]],
        [[1, 1], [0, 1], [0, 1]],
        [[0, 0, 1], [1, 1, 1]],
        [[1, 0], [1, 0], [1, 1]]
    ],
    # S tetromino - has 2 rotation states
    [
        [[0, 1, 1], [1, 1, 0]],
        [[1, 0], [1, 1], [0, 1]]
    ],
    # Z tetromino - has 2 rotation states
    [
        [[1, 1, 0], [0, 1, 1]],
        [[0, 1], [1, 1], [1, 0]]
    ]
]

# Colors for each tetromino
COLORS = [
    (0, 255, 255),   # Cyan (I)
    (255, 255, 0),   # Yellow (O)
    (128, 0, 128),   # Purple (T)
    (0, 0, 255),     # Blue (J)
    (255, 165, 0),   # Orange (L)
    (0, 255, 0),     # Green (S)
    (255, 0, 0)      # Red (Z)
]

# Actions: 0=left, 1=right, 2=rotate, 3=down, 4=drop, 5=do nothing
ACTION_LEFT = 0
ACTION_RIGHT = 1
ACTION_ROTATE = 2
ACTION_DOWN = 3
ACTION_DROP = 4
ACTION_NOTHING = 5

# Standard Tetris gravity intervals in milliseconds
# Based on the NES Tetris frames-per-line converted to milliseconds (assuming 60 FPS)
GRAVITY_INTERVALS = [
    800,    # Level 0: 48 frames ≈ 800ms
    717,    # Level 1: 43 frames ≈ 717ms
    633,    # Level 2: 38 frames ≈ 633ms
    550,    # Level 3: 33 frames ≈ 550ms
    467,    # Level 4: 28 frames ≈ 467ms
    383,    # Level 5: 23 frames ≈ 383ms
    300,    # Level 6: 18 frames ≈ 300ms
    217,    # Level 7: 13 frames ≈ 217ms
    133,    # Level 8: 8 frames ≈ 133ms
    100,    # Level 9: 6 frames ≈ 100ms
    83,     # Level 10: 5 frames ≈ 83ms
    83,     # Level 11: 5 frames
    83,     # Level 12: 5 frames
    67,     # Level 13: 4 frames ≈ 67ms
    67,     # Level 14: 4 frames
    67,     # Level 15: 4 frames
    50,     # Level 16: 3 frames ≈ 50ms
    50,     # Level 17: 3 frames
    50,     # Level 18: 3 frames
    33,     # Level 19: 2 frames ≈ 33ms
    33,     # Level 20: 2 frames
    33,     # Level 21: 2 frames
    33,     # Level 22: 2 frames
    33,     # Level 23: 2 frames
    33,     # Level 24: 2 frames
    33,     # Level 25: 2 frames
    33,     # Level 26: 2 frames
    33,     # Level 27: 2 frames
    33,     # Level 28: 2 frames
    17      # Level 29+: 1 frame ≈ 17ms
]

class SimpleTetrisEnv(gym.Env):
    """
    Simple Tetris environment that follows the gym interface.
    
    State representation:
    - A 2D grid of 0s and 1s (0=empty, 1=filled)
    - Current piece and position
    
    Action space:
    - 0: Move left
    - 1: Move right
    - 2: Rotate
    - 3: Move down
    - 4: Drop
    - 5: Do nothing
    """
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(self, grid_width=7, grid_height=14, render_mode=None):
        super(SimpleTetrisEnv, self).__init__()
        
        # Grid dimensions
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.grid = np.zeros((grid_height, grid_width), dtype=np.int32)
        
        # Game state
        self.score = 0
        self.lines_cleared = 0
        self.level = 1
        self.game_over = False
        self.steps_done = 0
        
        # Current piece state
        self.current_piece = None    # Index of current tetromino (0-6)
        self.current_rotation = 0    # Current rotation state
        self.current_x = 0           # X position of current piece
        self.current_y = 0           # Y position of current piece
        self.current_shape = None    # Current shape matrix
        self.current_shape_idx = 0   # Index for color reference
        
        # Next piece
        self.next_piece = None
        
        # Gravity mechanism - time-based for standard Tetris feel
        self.last_gravity_time = time.time()
        
        # Define action and observation spaces
        self.action_space = gym.spaces.Discrete(6)  # LEFT, RIGHT, ROTATE, DOWN, DROP, NOTHING
        
        # Observation space: grid + current piece + next piece
        self.observation_space = gym.spaces.Dict({
            'grid': gym.spaces.Box(low=0, high=7, shape=(grid_height, grid_width, 1), dtype=np.int32),
            'current_piece': gym.spaces.Discrete(7),
            'piece_x': gym.spaces.Discrete(grid_width),
            'piece_y': gym.spaces.Discrete(grid_height),
            'piece_rotation': gym.spaces.Discrete(4),
            'next_piece': gym.spaces.Discrete(7)
        })
        
        # Rendering setup
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.screen_width = grid_width * 30
        self.screen_height = grid_height * 30
        self.cell_size = 30
        
        # Delays for input handling
        self.das_delay = 0.15  # Delayed Auto Shift initial delay
        self.das_repeat = 0.05  # Delayed Auto Shift repeat rate
        self.last_key_time = 0

    def reset(self, seed=None):
        """Reset the environment to start a new game."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Reset the grid
        self.grid = np.zeros((self.grid_height, self.grid_width), dtype=np.int32)
        
        # Reset game state
        self.score = 0
        self.lines_cleared = 0
        self.level = 1
        self.game_over = False
        self.steps_done = 0
        
        # Reset piece state
        self.current_piece = None
        self.next_piece = random.randint(0, 6)  # Randomly select next piece
        
        # Reset gravity timer
        self.last_gravity_time = time.time()
        
        # Spawn the first piece
        self._spawn_piece()
        
        # Set up rendering if needed
        if self.render_mode == 'human' and self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption('Simple Tetris')
            self.clock = pygame.time.Clock()
        
        return self._get_observation()

    def _spawn_piece(self):
            """Spawn a new piece at the top of the grid."""
            if self.next_piece is None:
                self.next_piece = random.randint(0, 6)
                
            self.current_piece = self.next_piece
            self.current_shape_idx = self.current_piece
            self.next_piece = random.randint(0, 6)
            self.current_rotation = 0
            
            # Get the current shape based on piece and rotation
            self.current_shape = TETROMINO_SHAPES[self.current_piece][self.current_rotation]
            
            # Start position - center of the top row
            self.current_x = self.grid_width // 2 - len(self.current_shape[0]) // 2
            self.current_y = 0
            
            # Check if the new piece can be placed (game over condition)
            if not self._is_valid_position():
                # Print more detailed information about game over condition
                self.game_over = True
                # Check if there are blocks at the top rows
                top_filled = np.any(self.grid[0:2, :] > 0)
                # Print info if debugging
                if CONFIG.get("debug", False):
                    print(f"Game over detected. Top rows filled: {top_filled}")
                    print(f"Board height: {self._get_board_height()}")
                    print(f"Holes: {self._count_holes()}")

    def _is_valid_position(self):
        """Check if the current piece position is valid."""
        if self.current_shape is None:
            return False
        for y in range(len(self.current_shape)):
            for x in range(len(self.current_shape[y])):
                if self.current_shape[y][x]:
                    # Position check
                    if (self.current_y + y >= self.grid_height or 
                        self.current_x + x < 0 or 
                        self.current_x + x >= self.grid_width):
                        return False
                    # Collision check
                    if (self.current_y + y >= 0 and 
                        self.grid[self.current_y + y, self.current_x + x] > 0):
                        return False
        return True

    def _rotate(self):
        """Rotate the current piece clockwise and return if successful."""
        if self.current_piece is None:
            return False
        # Store original rotation
        original_rotation = self.current_rotation
        # Try to rotate
        new_rotation = (self.current_rotation + 1) % len(TETROMINO_SHAPES[self.current_piece])
        rotated_shape = TETROMINO_SHAPES[self.current_piece][new_rotation]
        # Save current shape
        old_shape = self.current_shape
        # Temporarily set the new rotation and shape
        self.current_rotation = new_rotation
        self.current_shape = rotated_shape
        # Check if rotation is valid (with wall kicks)
        for test_x_offset in [0, -1, 1, -2, 2]:
            test_x = self.current_x + test_x_offset
            old_x = self.current_x
            self.current_x = test_x
            if self._is_valid_position():
                return True
            self.current_x = old_x
        # Restore original rotation and shape if rotation fails
        self.current_rotation = original_rotation
        self.current_shape = old_shape
        return False

    def _move_left(self):
            """Try to move the current piece left."""
            self.current_x -= 1
            if not self._is_valid_position():
                self.current_x += 1
                return False
            return True

    def _move_right(self):
        """Try to move the current piece right."""
        self.current_x += 1
        if not self._is_valid_position():
            self.current_x -= 1
            return False
        return True

    def _move_down(self):
        """Try to move the current piece down."""
        self.current_y += 1
        if not self._is_valid_position():
            self.current_y -= 1
            return False
        return True

    def _place_piece(self):
        """Place the current piece on the grid and check for completed lines."""
        if self.game_over:
            return 0
        # Place the piece on the grid
        for y in range(len(self.current_shape)):
            for x in range(len(self.current_shape[y])):
                if self.current_shape[y][x]:
                    grid_y = self.current_y + y
                    grid_x = self.current_x + x
                    if 0 <= grid_y < self.grid_height and 0 <= grid_x < self.grid_width:
                        self.grid[grid_y, grid_x] = self.current_shape_idx + 1
        
        # Check for completed lines
        completed_lines = 0
        y = self.grid_height - 1
        
        # Debug full line detection
        if CONFIG.get("debug", False):
            for row_idx in range(self.grid_height):
                row = self.grid[row_idx, :]
                filled_cells = np.count_nonzero(row)
                if filled_cells >= self.grid_width * 0.75:  # Row is at least 75% full
                    print(f"Row {row_idx} is {filled_cells}/{self.grid_width} filled")
        
        # Process from bottom to top
        while y >= 0:
            # Check if row is completely filled
            if np.all(self.grid[y, :] > 0):
                # Move all lines above down
                for y2 in range(y, 0, -1):
                    self.grid[y2, :] = self.grid[y2-1, :]
                # Clear the top line
                self.grid[0, :] = 0
                completed_lines += 1
                # Don't decrement y since new line came down
            else:
                y -= 1
                
        if completed_lines > 0:
            self.lines_cleared += completed_lines
            self.score += self._calculate_score(completed_lines)
            self.level = self.lines_cleared // 10 + 1
            
            # Add explicit line clearing logging
            print(f"LINE CLEARED at step {self.steps_done} - agent cleared {completed_lines} lines!")
            
        self._spawn_piece()
        return completed_lines

    def _calculate_score(self, lines):
        """Calculate score for completed lines."""
        if lines == 1:
            return 40 * self.level
        elif lines == 2:
            return 100 * self.level
        elif lines == 3:
            return 300 * self.level
        elif lines == 4:
            return 1200 * self.level
        return 0

    def _get_board_height(self):
        """Calculate the current height of the board."""
        for y in range(self.grid_height):
            if np.any(self.grid[y, :] > 0):
                return self.grid_height - y
        return 0

    def _count_holes(self):
        """Count the number of holes in the grid.
        A hole is an empty cell with at least one filled cell above it.
        """
        holes = 0
        for col in range(self.grid_width):
            block_found = False
            for row in range(self.grid_height):
                if self.grid[row, col] > 0:
                    block_found = True
                elif block_found and self.grid[row, col] == 0:
                    holes += 1
        return holes
        
    def _calculate_bumpiness(self):
        """Calculate the bumpiness of the grid (sum of absolute differences between column heights)."""
        heights = []
        for col in range(self.grid_width):
            col_height = 0
            for row in range(self.grid_height):
                if self.grid[row, col] > 0:
                    col_height = self.grid_height - row
                    break
            heights.append(col_height)
        
        bumpiness = 0
        for i in range(len(heights) - 1):
            bumpiness += abs(heights[i] - heights[i + 1])
        
        return bumpiness

    def _get_observation(self):
        """Return the current observation."""
        observation = {
            'grid': self.grid.copy(),
            'current_piece': self.current_piece,
            'piece_x': self.current_x,
            'piece_y': self.current_y,
            'piece_rotation': self.current_rotation,
            'next_piece': self.next_piece
        }
        return observation

    def _measure_bumpiness(self):
        """
        Measure bumpiness of the board (sum of differences between adjacent columns).
        Same as _calculate_bumpiness but with a different name to match test expectations.
        """
        if hasattr(self, '_calculate_bumpiness'):
            return self._calculate_bumpiness()
        
        # If _calculate_bumpiness doesn't exist, implement it here
        bumpiness = 0
        heights = [0] * self.grid_width
        
        # Calculate the height of each column
        for x in range(self.grid_width):
            for y in range(self.grid_height):
                if self.grid[y, x] > 0:
                    heights[x] = self.grid_height - y
                    break
        
        # Calculate the sum of differences between adjacent columns
        for i in range(1, self.grid_width):
            bumpiness += abs(heights[i] - heights[i-1])
        
        return bumpiness

    def step(self, action):
        """
        Execute an action in the environment.
        (Improved Rewards: Line Clear + Height Penalty + Total Hole Penalty + Game Over)
        """
        reward = 0.0
        lines_cleared = 0
        info = {}
        self.steps_done += 1

        # Store previous hole count for delta calculation (we'll keep this for info)
        previous_holes = self._count_holes()
        previous_height = self._get_board_height()

        # Check for gravity
        current_time = time.time()
        force_gravity = False
        gravity_interval = GRAVITY_INTERVALS[min(self.level - 1, len(GRAVITY_INTERVALS) - 1)] / 1000.0
        if current_time - self.last_gravity_time > gravity_interval:
            force_gravity = True
            self.last_gravity_time = current_time

        # Process player action
        action_result = True
        placed_this_step = False
        if action == ACTION_LEFT:
            action_result = self._move_left()
        elif action == ACTION_RIGHT:
            action_result = self._move_right()
        elif action == ACTION_ROTATE:
            action_result = self._rotate()
        elif action == ACTION_DOWN:
            if not self._move_down():
                lines_cleared = self._place_piece()
                placed_this_step = True
        elif action == ACTION_DROP:
            while self._move_down():
                pass
            lines_cleared = self._place_piece()
            placed_this_step = True
        elif action == ACTION_NOTHING:
            pass

        info['action_result'] = action_result

        # Apply gravity
        if force_gravity and action != ACTION_DROP and not placed_this_step:
            if not self._move_down():
                if not placed_this_step: # Avoid double placement call
                    lines_cleared_gravity = self._place_piece()
                    # Use gravity lines cleared only if action didn't clear lines
                    if lines_cleared == 0:
                        lines_cleared = lines_cleared_gravity

        # --- Calculate Rewards ---
        line_clear_base_reward = 0.0
        if lines_cleared > 0:
            # Use the configured line clearing reward weight
            line_clear_base_reward = (lines_cleared ** 2) * CONFIG.get("reward_lines_cleared_weight", 1000.0)
            if lines_cleared == 4:
                line_clear_base_reward += CONFIG.get("reward_tetris_bonus", 2000.0) # Tetris bonus
            reward += line_clear_base_reward
            # Keep this one crucial print:
            print(f"@@@ LINE CLEAR REWARD: Base={line_clear_base_reward:.1f} for {lines_cleared} lines @@@")

        # Board height penalty
        board_height = self._get_board_height()
        height_penalty = board_height * CONFIG.get("reward_height_weight", -0.01)
        reward += height_penalty

        # FIXED: Total hole penalty (using absolute count instead of delta)
        current_holes = self._count_holes()
        holes_penalty = current_holes * CONFIG.get("reward_holes_weight", -0.1)
        reward += holes_penalty
        
        # Also calculate delta holes for info dict and debugging
        delta_holes = current_holes - previous_holes

        # Bumpiness penalty (if configured)
        if CONFIG.get("reward_bumpiness_weight", 0) != 0:
            bumpiness = self._measure_bumpiness()
            bumpiness_penalty = bumpiness * CONFIG.get("reward_bumpiness_weight", -0.01)
            reward += bumpiness_penalty
            info['bumpiness'] = bumpiness

        done = self.game_over
        if done:
            reward += CONFIG.get("reward_game_over_penalty", -10.0)

        # Final Scaling
        reward_scale = CONFIG.get("reward_scale", 1.0)
        final_reward = reward * reward_scale

        # --- Minimal Summary Print ---
        if CONFIG.get("debug", False):
            print(f"Step: {self.steps_done}, Action: {action}, Scaled Reward: {final_reward:.3f}, " +
                f"Lines: {lines_cleared}, Height: {board_height}, Holes: {current_holes} " +
                f"(Delta: {delta_holes}), Done: {done}")

        # Update info dictionary
        info['lines_cleared'] = lines_cleared
        info['board_height'] = board_height
        info['holes'] = current_holes
        info['delta_holes'] = delta_holes
        info['bumpiness'] = self._measure_bumpiness()
        info['score'] = self.score
        
        # Add potential lines info if available
        potential_lines = self._detect_potential_lines() if hasattr(self, '_detect_potential_lines') else 0
        info['potential_lines'] = potential_lines

        observation = self._get_observation()
        return observation, final_reward, done, info

    def _detect_potential_lines(self):
        """Detect rows that are close to being complete and Tetris opportunities."""
        potential_count = 0
        
        # Count nearly-complete rows (weighted by how close they are to completion)
        for row in range(self.grid_height):
            filled_cells = np.sum(self.grid[row, :] > 0)
            
            # Only count rows that are at least 60% filled (6 out of 10 cells for standard width)
            if filled_cells >= self.grid_width * 0.6:
                # Weight by how close to completion (0.1 to 1.0)
                completion_ratio = filled_cells / self.grid_width
                potential_count += completion_ratio * 0.5  # Reduce the weight to avoid inflated rewards
        
        # Look for Tetris setup opportunities (4 consecutive rows with high fill)
        tetris_setup_bonus = 0
        for base_row in range(self.grid_height - 3):  # Check 4-row sequences
            # Count filled cells in these 4 rows
            filled_per_row = [np.sum(self.grid[base_row + i, :] > 0) for i in range(4)]
            
            # Calculate average fill across all 4 rows
            avg_fill = sum(filled_per_row) / (4 * self.grid_width)
            
            # Only give Tetris setup bonus if average fill is high enough (70%+)
            if avg_fill >= 0.7:
                # Check if rows are arranged in a way that could lead to a Tetris
                # For example, if the bottom row has a gap in the middle that could be filled by an I piece
                bottom_row = self.grid[base_row + 3, :]
                if np.sum(bottom_row > 0) >= self.grid_width - 4 and np.any(bottom_row == 0):
                    tetris_setup_bonus = 2.0
                    break
                # If not a perfect Tetris setup, still give some bonus for high fill
                elif avg_fill >= 0.8:
                    tetris_setup_bonus = 1.0
                    break
        
        # Add the Tetris setup bonus to potential count
        potential_count += tetris_setup_bonus
        
        # Cap the potential lines reward to avoid excessive values
        potential_count = min(potential_count, 8.0)
        
        return potential_count

    def render(self, mode='human'):
        """Render the current game state."""
        if self.render_mode != 'human':
            return
        
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption('Simple Tetris')
            self.clock = pygame.time.Clock()
        
        self.screen.fill((0, 0, 0))
        
        # Draw grid cells
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                cell_value = self.grid[y, x]
                color = (40, 40, 40) if cell_value == 0 else COLORS[cell_value - 1]
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, (0, 0, 0), rect, 1)
        
        # Draw current piece
        if self.current_shape is not None:
            for y, row in enumerate(self.current_shape):
                for x, cell in enumerate(row):
                    if cell:
                        grid_x = self.current_x + x
                        grid_y = self.current_y + y
                        if grid_y >= 0:
                            color = COLORS[self.current_shape_idx]
                            rect = pygame.Rect(grid_x * self.cell_size, grid_y * self.cell_size, self.cell_size, self.cell_size)
                            pygame.draw.rect(self.screen, color, rect)
                            pygame.draw.rect(self.screen, (0, 0, 0), rect, 1)
        
        # Draw level and score information
        font = pygame.font.Font(None, 24)
        level_text = font.render(f"Level: {self.level}", True, (255, 255, 255))
        score_text = font.render(f"Score: {self.score}", True, (255, 255, 255))
        lines_text = font.render(f"Lines: {self.lines_cleared}", True, (255, 255, 255))
        
        # Position the text at the bottom of the screen
        self.screen.blit(level_text, (10, self.screen_height - 70))
        self.screen.blit(score_text, (10, self.screen_height - 45))
        self.screen.blit(lines_text, (10, self.screen_height - 20))
        
        pygame.display.flip()
        self.clock.tick(60)  # 60 FPS for smoother animation

    def close(self):
        """Clean up resources."""
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self.clock = None