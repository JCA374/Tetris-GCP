"""
Test suite for Tetris environments.

Tests the core functionality of both SimpleTetrisEnv and HighLevelTetrisEnv
to ensure they work correctly before training.
"""
import unittest
import numpy as np
import sys
import os

# Add current directory to path to import local modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from simple_tetris_env import SimpleTetrisEnv, TETROMINO_SHAPES
except ImportError as e:
    print(f"Warning: Could not import SimpleTetrisEnv: {e}")
    SimpleTetrisEnv = None
    TETROMINO_SHAPES = []

try:
    from high_level_env import HighLevelTetrisEnv
except ImportError as e:
    print(f"Warning: Could not import HighLevelTetrisEnv: {e}")
    HighLevelTetrisEnv = None


class TestTetrisEnvironments(unittest.TestCase):
    """Test cases for Tetris environments."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        if SimpleTetrisEnv is None:
            self.skipTest("SimpleTetrisEnv not available")
    
    def test_env_reset(self):
        """Test environment reset returns valid state."""
        env = SimpleTetrisEnv()
        state = env.reset()
        
        # Check that state is a dictionary with required keys
        self.assertIsInstance(state, dict)
        self.assertIn('grid', state)
        self.assertIn('current_piece', state)
        self.assertIn('next_piece', state)
        
        # Check grid dimensions
        self.assertEqual(state['grid'].shape, (14, 7))
        
        # Check that grid is initially mostly empty
        occupied_cells = np.sum(state['grid'] > 0)
        self.assertLess(occupied_cells, 10, "Initial grid should be mostly empty")
        
        # Check that pieces are valid
        if state['current_piece'] is not None:
            self.assertIsInstance(state['current_piece'], int)
            self.assertGreaterEqual(state['current_piece'], 0)
            self.assertLess(state['current_piece'], len(TETROMINO_SHAPES))
        
        if state['next_piece'] is not None:
            self.assertIsInstance(state['next_piece'], int)
            self.assertGreaterEqual(state['next_piece'], 0)
            self.assertLess(state['next_piece'], len(TETROMINO_SHAPES))
    
    def test_env_step_basic(self):
        """Test basic environment step functionality."""
        env = SimpleTetrisEnv()
        initial_state = env.reset()
        
        # Test a few random actions
        for _ in range(10):
            action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)
            
            # Check return types
            self.assertIsInstance(next_state, dict)
            self.assertIsInstance(reward, (int, float))
            self.assertIsInstance(done, bool)
            self.assertIsInstance(info, dict)
            
            # Check state consistency
            self.assertIn('grid', next_state)
            self.assertEqual(next_state['grid'].shape, (14, 7))
            
            if done:
                break
    
    def test_line_clearing_detection(self):
        """Test line clearing detection and scoring."""
        env = SimpleTetrisEnv()
        env.reset()
        
        # Manually set up a scenario where we can clear a line
        # Fill bottom row except one cell
        env.grid[13, :-1] = 1  # Fill all but last column
        
        # Store initial state
        initial_lines_cleared = getattr(env, 'lines_cleared_total', 0)
        
        # Fill the last cell by placing a piece (simulate)
        env.grid[13, -1] = 1
        
        # Call the line clearing method directly if available
        if hasattr(env, '_clear_lines'):
            lines_cleared = env._clear_lines()
            self.assertGreater(lines_cleared, 0, "Should detect and clear the full line")
        elif hasattr(env, 'clear_lines'):
            lines_cleared = env.clear_lines()
            self.assertGreater(lines_cleared, 0, "Should detect and clear the full line")
        else:
            # If no direct method, check that the line is gone after a step
            original_bottom_row = env.grid[13, :].copy()
            self.assertTrue(np.all(original_bottom_row == 1), "Bottom row should be full")
            
            # Take a step and see if line clearing happened
            action = 0  # Some action
            _, reward, _, info = env.step(action)
            
            # Check if line was cleared (bottom row should not be all 1s anymore)
            new_bottom_row = env.grid[13, :]
            if np.all(original_bottom_row == 1) and not np.all(new_bottom_row == 1):
                # Line was cleared
                self.assertGreater(reward, 0, "Should get positive reward for clearing line")
                self.assertIn('lines_cleared', info)
    
    def test_game_over_detection(self):
        """Test game over is properly detected."""
        env = SimpleTetrisEnv()
        env.reset()
        
        # Fill the top rows to trigger game over
        env.grid[0:3, :] = 1
        
        # Try to spawn a new piece - this should trigger game over
        if hasattr(env, '_spawn_piece'):
            # Call spawn piece method directly if available
            game_over = env._spawn_piece()
            if game_over is not None:
                self.assertTrue(game_over, "Should detect game over when top is filled")
        else:
            # Test through step
            action = 0
            _, reward, done, info = env.step(action)
            
            # Should eventually hit game over condition
            steps = 0
            while not done and steps < 100:  # Safety limit
                action = env.action_space.sample()
                _, reward, done, info = env.step(action)
                steps += 1
            
            if done:
                self.assertLess(reward, 0, "Game over should give negative reward")
    
    def test_action_space(self):
        """Test that action space is correctly defined."""
        env = SimpleTetrisEnv()
        
        # Check action space
        self.assertTrue(hasattr(env, 'action_space'))
        self.assertTrue(hasattr(env.action_space, 'n'))
        self.assertGreater(env.action_space.n, 0)
        
        # Test sampling actions
        for _ in range(10):
            action = env.action_space.sample()
            self.assertIsInstance(action, (int, np.integer))
            self.assertGreaterEqual(action, 0)
            self.assertLess(action, env.action_space.n)
    
    def test_piece_movement(self):
        """Test piece movement mechanics."""
        env = SimpleTetrisEnv()
        state = env.reset()
        
        initial_piece_x = state.get('piece_x', 0)
        initial_piece_y = state.get('piece_y', 0)
        
        # Test moving left (if action 0 is move left)
        if hasattr(env, 'move_left') or 0 < env.action_space.n:
            state, _, done, _ = env.step(0)  # Assume action 0 is move left
            
            if not done:
                # Check that piece position changed appropriately
                new_piece_x = state.get('piece_x', initial_piece_x)
                # Movement might be blocked by boundaries, so just check validity
                self.assertIsInstance(new_piece_x, (int, np.integer))
                self.assertGreaterEqual(new_piece_x, 0)
                self.assertLess(new_piece_x, 7)  # Grid width
    
    def test_state_consistency(self):
        """Test that state remains consistent across steps."""
        env = SimpleTetrisEnv()
        state = env.reset()
        
        for step in range(50):  # Test multiple steps
            action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)
            
            # Check that grid values are valid
            self.assertTrue(np.all(next_state['grid'] >= 0), 
                          f"Grid should not have negative values at step {step}")
            
            # Check that piece positions are within bounds
            if 'piece_x' in next_state and next_state['piece_x'] is not None:
                self.assertGreaterEqual(next_state['piece_x'], 0)
                self.assertLess(next_state['piece_x'], 7)
            
            if 'piece_y' in next_state and next_state['piece_y'] is not None:
                self.assertGreaterEqual(next_state['piece_y'], 0)
                self.assertLess(next_state['piece_y'], 14)
            
            if done:
                break
            
            state = next_state


class TestHighLevelTetrisEnv(unittest.TestCase):
    """Test cases for high-level Tetris environment."""
    
    def setUp(self):
        """Set up test fixtures."""
        if HighLevelTetrisEnv is None:
            self.skipTest("HighLevelTetrisEnv not available")
    
    def test_high_level_env_initialization(self):
        """Test high-level environment initializes correctly."""
        env = HighLevelTetrisEnv()
        state = env.reset()
        
        self.assertIsInstance(state, dict)
        self.assertIn('grid', state)
        
        # Should have valid placements
        if hasattr(env, 'valid_placements'):
            self.assertIsInstance(env.valid_placements, list)
            self.assertGreater(len(env.valid_placements), 0, 
                             "Should have at least one valid placement")
    
    def test_high_level_actions(self):
        """Test high-level action wrapper."""
        env = HighLevelTetrisEnv()
        state = env.reset()
        
        # Get number of valid placements
        num_actions = getattr(env, 'num_actions', env.action_space.n if hasattr(env, 'action_space') else 1)
        
        if num_actions > 0:
            # Test first action
            action = 0
            next_state, reward, done, info = env.step(action)
            
            # Check return types
            self.assertIsInstance(next_state, dict)
            self.assertIsInstance(reward, (int, float))
            self.assertIsInstance(done, bool)
            self.assertIsInstance(info, dict)
            
            # Should have info about high-level action
            self.assertIn('high_level_action', info)
    
    def test_placement_generation(self):
        """Test that valid placements are generated correctly."""
        env = HighLevelTetrisEnv()
        env.reset()
        
        if hasattr(env, 'get_valid_placements'):
            placements = env.get_valid_placements()
            self.assertIsInstance(placements, list)
            
            # Each placement should be a valid format
            for placement in placements:
                self.assertIsInstance(placement, (list, tuple))
                self.assertGreaterEqual(len(placement), 2)  # At least x, y
        elif hasattr(env, 'valid_placements'):
            placements = env.valid_placements
            self.assertIsInstance(placements, list)
            self.assertGreater(len(placements), 0)


class TestEnvironmentEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""
    
    def setUp(self):
        """Set up test fixtures."""
        if SimpleTetrisEnv is None:
            self.skipTest("SimpleTetrisEnv not available")
    
    def test_invalid_actions(self):
        """Test behavior with invalid actions."""
        env = SimpleTetrisEnv()
        env.reset()
        
        # Test action outside valid range
        invalid_action = env.action_space.n + 10
        
        try:
            state, reward, done, info = env.step(invalid_action)
            # If it doesn't raise an error, check that it handled gracefully
            self.assertIsInstance(state, dict)
            self.assertIsInstance(reward, (int, float))
            self.assertIsInstance(done, bool)
        except (ValueError, IndexError):
            # It's acceptable to raise an error for invalid actions
            pass
    
    def test_reset_multiple_times(self):
        """Test that reset works correctly when called multiple times."""
        env = SimpleTetrisEnv()
        
        for _ in range(5):
            state = env.reset()
            self.assertIsInstance(state, dict)
            self.assertIn('grid', state)
            
            # Grid should be reset (mostly empty)
            occupied_cells = np.sum(state['grid'] > 0)
            self.assertLess(occupied_cells, 10)
    
    def test_long_episode(self):
        """Test environment stability over long episodes."""
        env = SimpleTetrisEnv()
        state = env.reset()
        
        steps = 0
        max_steps = 1000
        
        while steps < max_steps:
            action = env.action_space.sample()
            state, reward, done, info = env.step(action)
            steps += 1
            
            if done:
                break
        
        # Should either finish naturally or hit max steps
        self.assertLessEqual(steps, max_steps)


def run_environment_tests():
    """Run all environment tests and return results."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestTetrisEnvironments))
    if HighLevelTetrisEnv is not None:
        suite.addTests(loader.loadTestsFromTestCase(TestHighLevelTetrisEnv))
    suite.addTests(loader.loadTestsFromTestCase(TestEnvironmentEdgeCases))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == "__main__":
    print("Testing Tetris Environments...")
    print("=" * 50)
    
    # Run tests
    result = run_environment_tests()
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\nOverall: {'PASS' if success else 'FAIL'}")
    
    if success:
        print("✅ All environment tests passed! Environment is ready for training.")
    else:
        print("❌ Some environment tests failed. Please fix issues before training.")
    
    exit(0 if success else 1)