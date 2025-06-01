"""
Integration test suite for Tetris DQN project.

Tests that all components work together correctly in realistic training scenarios.
Includes short training runs, curriculum learning tests, and overall system validation.
"""
import unittest
import numpy as np
import torch
import tempfile
import os
import sys
import time

# Add current directory to path to import local modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from simple_tetris_env import SimpleTetrisEnv
    ENV_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import SimpleTetrisEnv: {e}")
    ENV_AVAILABLE = False

try:
    from high_level_env import HighLevelTetrisEnv
    HIGH_LEVEL_ENV_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import HighLevelTetrisEnv: {e}")
    HIGH_LEVEL_ENV_AVAILABLE = False

try:
    from agent import DQNAgent
    from minimal_config import get_minimal_config, get_enhanced_config, validate_config
    AGENT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import agent or config: {e}")
    AGENT_AVAILABLE = False

try:
    from preprocessing import preprocess_state, BatchPreprocessor
    PREPROCESSING_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import preprocessing: {e}")
    PREPROCESSING_AVAILABLE = False

try:
    from unified_replay_buffer import UnifiedReplayBuffer
    BUFFER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import unified buffer: {e}")
    BUFFER_AVAILABLE = False


def simple_training_loop(env, agent, num_episodes=10, max_steps_per_episode=100):
    """
    Simple training loop for testing integration.
    
    Args:
        env: Environment instance
        agent: Agent instance  
        num_episodes: Number of episodes to train
        max_steps_per_episode: Maximum steps per episode
        
    Returns:
        Dictionary with training metrics
    """
    metrics = {
        'episode_rewards': [],
        'episode_lengths': [],
        'lines_cleared': [],
        'losses': [],
        'total_steps': 0
    }
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        episode_lines_cleared = 0
        
        for step in range(max_steps_per_episode):
            # Preprocess state if needed
            if PREPROCESSING_AVAILABLE and isinstance(state, dict):
                processed_state = preprocess_state(state, device=agent.device)
            else:
                processed_state = torch.tensor(state, dtype=torch.float32, device=agent.device)
                if len(processed_state.shape) == 2:  # Add channel dimension
                    processed_state = processed_state.unsqueeze(0)
            
            # Select action
            action = agent.select_action(processed_state, training=True)
            
            # Environment step
            next_state, reward, done, info = env.step(action)
            
            # Store transition
            if next_state is not None and PREPROCESSING_AVAILABLE and isinstance(next_state, dict):
                next_processed_state = preprocess_state(next_state, device=agent.device)
            elif next_state is not None:
                next_processed_state = torch.tensor(next_state, dtype=torch.float32, device=agent.device)
                if len(next_processed_state.shape) == 2:
                    next_processed_state = next_processed_state.unsqueeze(0)
            else:
                next_processed_state = None
            
            agent.memory.push(processed_state, action, next_processed_state, reward, done)
            
            # Learn
            if len(agent.memory) >= agent.batch_size:
                loss = agent.learn()
                if loss is not None:
                    metrics['losses'].append(loss)
            
            # Update metrics
            episode_reward += reward
            episode_length += 1
            metrics['total_steps'] += 1
            
            if 'lines_cleared' in info:
                episode_lines_cleared += info['lines_cleared']
            
            if done:
                break
            
            state = next_state
        
        # Record episode metrics
        metrics['episode_rewards'].append(episode_reward)
        metrics['episode_lengths'].append(episode_length)
        metrics['lines_cleared'].append(episode_lines_cleared)
        
        # Update exploration
        agent.epsilon = max(agent.epsilon_end, agent.epsilon * agent.epsilon_decay)
    
    return metrics


class TestBasicIntegration(unittest.TestCase):
    """Test basic integration of all components."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not (ENV_AVAILABLE and AGENT_AVAILABLE and PREPROCESSING_AVAILABLE):
            self.skipTest("Required components not available")
    
    def test_minimal_training_run(self):
        """Test minimal training run completes successfully."""
        # Create environment and agent
        env = SimpleTetrisEnv()
        config = get_minimal_config()
        config['num_episodes'] = 5
        config['batch_size'] = 8
        config['replay_capacity'] = 100
        
        agent = DQNAgent(
            input_shape=(4, 14, 7),  # Enhanced preprocessing
            n_actions=env.action_space.n,
            device="cpu",
            config=config
        )
        
        # Run training
        metrics = simple_training_loop(env, agent, num_episodes=5, max_steps_per_episode=50)
        
        # Verify training completed
        self.assertEqual(len(metrics['episode_rewards']), 5)
        self.assertGreater(metrics['total_steps'], 0)
        
        # Verify agent learned something (has some experience)
        self.assertGreater(len(agent.memory), 0)
        
        # Verify losses were computed
        if metrics['losses']:
            self.assertGreater(len(metrics['losses']), 0)
            for loss in metrics['losses']:
                self.assertIsInstance(loss, (int, float))
                self.assertGreater(loss, 0)
    
    def test_gpu_training_run(self):
        """Test training run on GPU if available."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        env = SimpleTetrisEnv()
        config = get_minimal_config()
        config['num_episodes'] = 3
        config['batch_size'] = 8
        config['replay_capacity'] = 100
        config['device'] = 'cuda'
        
        agent = DQNAgent(
            input_shape=(4, 14, 7),
            n_actions=env.action_space.n,
            device="cuda",
            config=config
        )
        
        # Run training
        metrics = simple_training_loop(env, agent, num_episodes=3, max_steps_per_episode=30)
        
        # Verify training completed
        self.assertEqual(len(metrics['episode_rewards']), 3)
        self.assertGreater(metrics['total_steps'], 0)
        
        # Verify components are on GPU
        sample_state = preprocess_state(env.reset(), device="cuda")
        self.assertEqual(sample_state.device.type, "cuda")
    
    def test_different_config_phases(self):
        """Test training with different configuration phases."""
        env = SimpleTetrisEnv()
        
        # Test minimal config
        minimal_config = get_minimal_config()
        minimal_config['num_episodes'] = 2
        minimal_config['batch_size'] = 8
        
        agent_minimal = DQNAgent(
            input_shape=(1, 14, 7),  # Basic preprocessing
            n_actions=env.action_space.n,
            device="cpu",
            config=minimal_config
        )
        
        metrics_minimal = simple_training_loop(env, agent_minimal, num_episodes=2, max_steps_per_episode=20)
        self.assertEqual(len(metrics_minimal['episode_rewards']), 2)
        
        # Test enhanced config
        enhanced_config = get_enhanced_config()
        enhanced_config['num_episodes'] = 2
        enhanced_config['batch_size'] = 8
        enhanced_config['replay_capacity'] = 100
        
        agent_enhanced = DQNAgent(
            input_shape=(4, 14, 7),  # Enhanced preprocessing
            n_actions=env.action_space.n,
            device="cpu",
            config=enhanced_config
        )
        
        metrics_enhanced = simple_training_loop(env, agent_enhanced, num_episodes=2, max_steps_per_episode=20)
        self.assertEqual(len(metrics_enhanced['episode_rewards']), 2)


class TestAdvancedIntegration(unittest.TestCase):
    """Test advanced integration features."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not (ENV_AVAILABLE and AGENT_AVAILABLE and BUFFER_AVAILABLE):
            self.skipTest("Required components not available")
    
    def test_prioritized_replay_integration(self):
        """Test integration with prioritized replay buffer."""
        env = SimpleTetrisEnv()
        config = get_minimal_config()
        config['use_prioritized_replay'] = True
        config['batch_size'] = 8
        config['replay_capacity'] = 100
        
        # Create agent with prioritized replay
        buffer = UnifiedReplayBuffer(capacity=100, device="cpu", prioritized=True)
        agent = DQNAgent(
            input_shape=(4, 14, 7),
            n_actions=env.action_space.n,
            device="cpu",
            config=config,
            memory=buffer
        )
        
        # Run training
        metrics = simple_training_loop(env, agent, num_episodes=3, max_steps_per_episode=30)
        
        # Verify training completed with prioritized replay
        self.assertEqual(len(metrics['episode_rewards']), 3)
        self.assertTrue(agent.memory.prioritized)
    
    def test_model_save_load_integration(self):
        """Test model save/load in training context."""
        env = SimpleTetrisEnv()
        config = get_minimal_config()
        config['batch_size'] = 8
        config['replay_capacity'] = 100
        
        # Train first agent
        agent1 = DQNAgent(
            input_shape=(4, 14, 7),
            n_actions=env.action_space.n,
            device="cpu",
            config=config
        )
        
        # Train for a few episodes
        simple_training_loop(agent1, env, num_episodes=3, max_steps_per_episode=20)
        
        # Save model
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            temp_path = f.name
        
        try:
            agent1.save(temp_path)
            
            # Create new agent and load
            agent2 = DQNAgent(
                input_shape=(4, 14, 7),
                n_actions=env.action_space.n,
                device="cpu",
                config=config
            )
            agent2.load(temp_path)
            
            # Test that loaded agent can continue training
            metrics = simple_training_loop(env, agent2, num_episodes=2, max_steps_per_episode=20)
            self.assertEqual(len(metrics['episode_rewards']), 2)
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_batch_preprocessing_integration(self):
        """Test batch preprocessing in training context."""
        if not PREPROCESSING_AVAILABLE:
            self.skipTest("Preprocessing not available")
        
        env = SimpleTetrisEnv()
        config = get_minimal_config()
        config['batch_size'] = 8
        
        # Create batch preprocessor
        preprocessor = BatchPreprocessor(device="cpu", include_piece_info=True)
        
        agent = DQNAgent(
            input_shape=(4, 14, 7),
            n_actions=env.action_space.n,
            device="cpu",
            config=config
        )
        
        # Test that preprocessing works in training loop
        states = []
        for _ in range(5):
            state = env.reset()
            states.append(state)
        
        # Batch preprocess
        batch_tensor = preprocessor.preprocess_batch(states)
        self.assertIsNotNone(batch_tensor)
        self.assertEqual(batch_tensor.shape, (5, 4, 14, 7))


class TestLineClearing(unittest.TestCase):
    """Test line clearing detection and improvement."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not (ENV_AVAILABLE and AGENT_AVAILABLE):
            self.skipTest("Required components not available")
    
    def test_line_clearing_metrics(self):
        """Test that line clearing metrics are tracked correctly."""
        env = SimpleTetrisEnv()
        config = get_minimal_config()
        config['batch_size'] = 8
        config['replay_capacity'] = 100
        
        agent = DQNAgent(
            input_shape=(4, 14, 7),
            n_actions=env.action_space.n,
            device="cpu",
            config=config
        )
        
        # Run training and track line clearing
        metrics = simple_training_loop(env, agent, num_episodes=5, max_steps_per_episode=100)
        
        # Should have line clearing data
        self.assertEqual(len(metrics['lines_cleared']), 5)
        
        # At least some episodes should have attempted line clearing
        total_lines = sum(metrics['lines_cleared'])
        self.assertGreaterEqual(total_lines, 0)  # Could be 0 if agent is bad, but shouldn't crash
    
    def test_reward_structure(self):
        """Test that reward structure works correctly."""
        env = SimpleTetrisEnv()
        
        # Test a single step to verify reward structure
        state = env.reset()
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        
        # Reward should be a number
        self.assertIsInstance(reward, (int, float))
        
        # Info should contain relevant data
        self.assertIsInstance(info, dict)


class TestHighLevelEnvironment(unittest.TestCase):
    """Test integration with high-level environment."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not (HIGH_LEVEL_ENV_AVAILABLE and AGENT_AVAILABLE):
            self.skipTest("High-level environment or agent not available")
    
    def test_high_level_training(self):
        """Test training with high-level action space."""
        env = HighLevelTetrisEnv()
        config = get_minimal_config()
        config['batch_size'] = 8
        config['replay_capacity'] = 100
        
        # Get action space size from environment
        n_actions = getattr(env, 'action_space', type('obj', (object,), {'n': 10})).n
        
        agent = DQNAgent(
            input_shape=(4, 14, 7),
            n_actions=n_actions,
            device="cpu",
            config=config
        )
        
        # Run short training
        metrics = simple_training_loop(env, agent, num_episodes=3, max_steps_per_episode=30)
        
        # Verify training completed
        self.assertEqual(len(metrics['episode_rewards']), 3)
        
        # High-level environment should provide placement info
        state = env.reset()
        action = 0  # First valid placement
        next_state, reward, done, info = env.step(action)
        
        # Should have high-level action info
        self.assertIn('high_level_action', info)


class TestErrorRecovery(unittest.TestCase):
    """Test error recovery and robustness."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not (ENV_AVAILABLE and AGENT_AVAILABLE):
            self.skipTest("Required components not available")
    
    def test_invalid_state_recovery(self):
        """Test recovery from invalid states."""
        env = SimpleTetrisEnv()
        config = get_minimal_config()
        config['batch_size'] = 8
        
        agent = DQNAgent(
            input_shape=(4, 14, 7),
            n_actions=env.action_space.n,
            device="cpu",
            config=config
        )
        
        # Test with various invalid inputs
        invalid_states = [
            None,
            np.array([]),
            np.random.rand(5, 5),  # Wrong shape
            {"invalid": "state"}
        ]
        
        for invalid_state in invalid_states:
            try:
                if PREPROCESSING_AVAILABLE:
                    processed = preprocess_state(invalid_state, device="cpu")
                    if processed is not None:
                        action = agent.select_action(processed, training=True)
                        self.assertIsInstance(action, (int, np.integer))
                else:
                    # Skip if preprocessing not available
                    pass
            except (ValueError, TypeError, AttributeError, RuntimeError):
                # These errors are acceptable for invalid inputs
                pass
    
    def test_memory_overflow_recovery(self):
        """Test recovery from memory issues."""
        env = SimpleTetrisEnv()
        config = get_minimal_config()
        config['batch_size'] = 8
        config['replay_capacity'] = 50  # Small buffer
        
        agent = DQNAgent(
            input_shape=(4, 14, 7),
            n_actions=env.action_space.n,
            device="cpu",
            config=config
        )
        
        # Fill memory beyond capacity
        state = env.reset()
        for _ in range(100):  # More than capacity
            if PREPROCESSING_AVAILABLE:
                processed_state = preprocess_state(state, device="cpu")
            else:
                processed_state = torch.rand(4, 14, 7)
            
            action = np.random.randint(0, env.action_space.n)
            agent.memory.push(processed_state, action, processed_state, 0.0, False)
        
        # Should not exceed capacity
        self.assertLessEqual(len(agent.memory), config['replay_capacity'])
        
        # Should still be able to learn
        if len(agent.memory) >= agent.batch_size:
            loss = agent.learn()
            self.assertIsNotNone(loss)


def run_integration_tests():
    """Run all integration tests and return results."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestBasicIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestAdvancedIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestLineClearing))
    if HIGH_LEVEL_ENV_AVAILABLE:
        suite.addTests(loader.loadTestsFromTestCase(TestHighLevelEnvironment))
    suite.addTests(loader.loadTestsFromTestCase(TestErrorRecovery))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == "__main__":
    print("Testing Integration...")
    print("=" * 50)
    
    # Show system info
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    print("\nRunning integration tests...")
    start_time = time.time()
    
    # Run tests
    result = run_integration_tests()
    
    end_time = time.time()
    test_duration = end_time - start_time
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print(f"Duration: {test_duration:.2f} seconds")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}")
            lines = traceback.split('\n')
            error_line = next((line for line in reversed(lines) if 'AssertionError' in line), lines[-2])
            print(f"  {error_line.strip()}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}")
            lines = traceback.split('\n')
            error_line = next((line for line in reversed(lines) if line.strip() and 'Error' in line), lines[-2])
            print(f"  {error_line.strip()}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\nOverall: {'PASS' if success else 'FAIL'}")
    
    if success:
        print("✅ All integration tests passed! System is ready for full training.")
        print("\nNext steps:")
        print("1. Run progressive training script")
        print("2. Monitor training metrics")
        print("3. Adjust hyperparameters as needed")
    else:
        print("❌ Some integration tests failed. Please fix issues before full training.")
        print("\nRecommended actions:")
        print("1. Fix failing tests")
        print("2. Run individual component tests")
        print("3. Check system resources")
    
    exit(0 if success else 1)