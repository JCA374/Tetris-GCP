"""
Test suite for DQN Agent functionality.

Tests the DQN agent to ensure learning, action selection, and model 
save/load functionality work correctly.
"""
import unittest
import numpy as np
import torch
import tempfile
import os
import sys

# Add current directory to path to import local modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from agent import DQNAgent
    from minimal_config import get_minimal_config
    AGENT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import agent module: {e}")
    AGENT_AVAILABLE = False

try:
    from unified_replay_buffer import UnifiedReplayBuffer
    UNIFIED_BUFFER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import unified replay buffer: {e}")
    UNIFIED_BUFFER_AVAILABLE = False

try:
    from model import DQN, DuelDQN
    MODEL_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import model: {e}")
    MODEL_AVAILABLE = False


def create_test_agent(config=None, device="cpu"):
    """Create a test agent with minimal configuration."""
    if not AGENT_AVAILABLE:
        return None
    
    if config is None:
        config = get_minimal_config()
        config["device"] = device
        config["replay_capacity"] = 1000  # Small for testing
        config["batch_size"] = 16  # Small for testing
    
    input_shape = (1, 14, 7) if not config.get("use_enhanced_preprocessing", False) else (4, 14, 7)
    n_actions = 7  # Standard Tetris actions
    
    agent = DQNAgent(
        input_shape=input_shape,
        n_actions=n_actions,
        device=device,
        config=config
    )
    
    return agent


def create_test_state(enhanced=False, device="cpu"):
    """Create a test state for agent testing."""
    if enhanced:
        state = torch.rand(4, 14, 7, device=device)
    else:
        state = torch.rand(1, 14, 7, device=device)
    return state


def fill_replay_buffer(agent, num_samples=100):
    """Fill agent's replay buffer with random transitions."""
    for _ in range(num_samples):
        state = create_test_state(device=agent.device)
        action = np.random.randint(0, agent.n_actions)
        next_state = create_test_state(device=agent.device)
        reward = np.random.randn()
        done = np.random.random() < 0.1  # 10% chance of done
        
        agent.memory.push(state, action, next_state, reward, done)


class TestDQNAgent(unittest.TestCase):
    """Test cases for DQN Agent."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not AGENT_AVAILABLE:
            self.skipTest("Agent module not available")
    
    def test_agent_initialization(self):
        """Test agent initializes correctly."""
        agent = create_test_agent()
        
        # Check basic attributes
        self.assertIsNotNone(agent.policy_net)
        self.assertIsNotNone(agent.target_net)
        self.assertIsNotNone(agent.optimizer)
        self.assertIsNotNone(agent.memory)
        
        # Check configuration
        self.assertEqual(agent.device, "cpu")
        self.assertGreater(agent.learning_rate, 0)
        self.assertGreater(agent.batch_size, 0)
    
    def test_agent_device_placement(self):
        """Test agent components are on correct device."""
        if torch.cuda.is_available():
            agent = create_test_agent(device="cuda")
            
            # Check networks are on GPU
            self.assertEqual(next(agent.policy_net.parameters()).device.type, "cuda")
            self.assertEqual(next(agent.target_net.parameters()).device.type, "cuda")
        
        # CPU test
        agent_cpu = create_test_agent(device="cpu")
        self.assertEqual(next(agent_cpu.policy_net.parameters()).device.type, "cpu")
    
    def test_action_selection_deterministic(self):
        """Test deterministic action selection."""
        agent = create_test_agent()
        
        # Set epsilon to 0 for deterministic behavior
        agent.epsilon = 0.0
        
        state = create_test_state(device=agent.device)
        
        # Multiple calls should return same action
        actions = [agent.select_action(state, training=True) for _ in range(10)]
        
        # All actions should be the same (deterministic)
        self.assertEqual(len(set(actions)), 1, "Deterministic selection should return same action")
        
        # Action should be valid
        action = actions[0]
        self.assertIsInstance(action, (int, np.integer))
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, agent.n_actions)
    
    def test_action_selection_exploration(self):
        """Test exploration in action selection."""
        agent = create_test_agent()
        
        # Set epsilon to 1 for maximum exploration
        agent.epsilon = 1.0
        
        state = create_test_state(device=agent.device)
        
        # Multiple calls should return different actions (with high probability)
        actions = [agent.select_action(state, training=True) for _ in range(100)]
        
        # Should have some variety in actions
        unique_actions = len(set(actions))
        self.assertGreater(unique_actions, 1, "Exploration should produce different actions")
        
        # All actions should be valid
        for action in actions:
            self.assertIsInstance(action, (int, np.integer))
            self.assertGreaterEqual(action, 0)
            self.assertLess(action, agent.n_actions)
    
    def test_action_selection_eval_mode(self):
        """Test action selection in evaluation mode."""
        agent = create_test_agent()
        agent.epsilon = 1.0  # High exploration in training
        
        state = create_test_state(device=agent.device)
        
        # Evaluation mode should be deterministic regardless of epsilon
        actions = [agent.select_action(state, training=False) for _ in range(10)]
        self.assertEqual(len(set(actions)), 1, "Eval mode should be deterministic")
    
    def test_learning_step(self):
        """Test single learning step executes without error."""
        agent = create_test_agent()
        
        # Fill replay buffer
        fill_replay_buffer(agent, 100)
        
        # Test learning
        loss = agent.learn()
        
        self.assertIsNotNone(loss, "Learning should return a loss value")
        self.assertIsInstance(loss, (int, float))
        self.assertTrue(0 <= loss <= 1000, f"Loss should be reasonable, got {loss}")
    
    def test_learning_insufficient_data(self):
        """Test learning behavior with insufficient data."""
        agent = create_test_agent()
        
        # Don't fill buffer enough
        fill_replay_buffer(agent, 5)  # Less than batch_size
        
        # Learning should return None or handle gracefully
        loss = agent.learn()
        self.assertIsNone(loss, "Learning with insufficient data should return None")
    
    def test_memory_storage(self):
        """Test that experiences are stored correctly."""
        agent = create_test_agent()
        
        initial_size = len(agent.memory)
        
        # Add some experiences
        for _ in range(10):
            state = create_test_state(device=agent.device)
            action = np.random.randint(0, agent.n_actions)
            next_state = create_test_state(device=agent.device)
            reward = np.random.randn()
            done = False
            
            agent.memory.push(state, action, next_state, reward, done)
        
        final_size = len(agent.memory)
        self.assertEqual(final_size - initial_size, 10, "Memory should store all experiences")
    
    def test_target_network_update(self):
        """Test target network update functionality."""
        agent = create_test_agent()
        
        # Get initial target network weights
        initial_target_params = [p.clone() for p in agent.target_net.parameters()]
        
        # Update policy network (simulate training)
        fill_replay_buffer(agent, 100)
        for _ in range(5):
            agent.learn()
        
        # Update target network
        agent.update_target_network()
        
        # Check that target network changed
        updated_target_params = list(agent.target_net.parameters())
        
        # At least one parameter should have changed
        params_changed = any(
            not torch.equal(initial, updated) 
            for initial, updated in zip(initial_target_params, updated_target_params)
        )
        self.assertTrue(params_changed, "Target network should update")
    
    def test_model_save_load(self):
        """Test model can be saved and loaded correctly."""
        agent1 = create_test_agent()
        
        # Train briefly to change weights
        fill_replay_buffer(agent1, 100)
        for _ in range(10):
            agent1.learn()
        
        # Save model
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            temp_path = f.name
        
        try:
            agent1.save(temp_path)
            
            # Create new agent and load
            agent2 = create_test_agent()
            agent2.load(temp_path)
            
            # Compare predictions
            state = create_test_state(device=agent1.device)
            
            with torch.no_grad():
                q_values1 = agent1.policy_net(state.unsqueeze(0))
                q_values2 = agent2.policy_net(state.unsqueeze(0))
            
            # Q-values should be nearly identical
            self.assertTrue(torch.allclose(q_values1, q_values2, atol=1e-6))
            
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_epsilon_decay(self):
        """Test epsilon decay functionality."""
        config = get_minimal_config()
        config["epsilon_decay"] = 0.9  # Fast decay for testing
        agent = create_test_agent(config=config)
        
        initial_epsilon = agent.epsilon
        
        # Simulate multiple steps
        state = create_test_state(device=agent.device)
        for _ in range(10):
            agent.select_action(state, training=True)
        
        # Epsilon should have decayed
        self.assertLess(agent.epsilon, initial_epsilon, "Epsilon should decay over time")
        self.assertGreaterEqual(agent.epsilon, agent.epsilon_end, "Epsilon should not go below minimum")
    
    def test_learning_rate_warmup(self):
        """Test learning rate warmup functionality."""
        config = get_minimal_config()
        config["lr_warmup_steps"] = 10
        config["lr_warmup_lr"] = 1e-6
        agent = create_test_agent(config=config)
        
        # Initially should be at warmup LR
        current_lr = agent.optimizer.param_groups[0]['lr']
        self.assertAlmostEqual(current_lr, 1e-6, places=7)
        
        # After learning steps, LR should increase
        fill_replay_buffer(agent, 100)
        for _ in range(15):  # More than warmup steps
            agent.learn()
        
        final_lr = agent.optimizer.param_groups[0]['lr']
        self.assertGreater(final_lr, 1e-6, "LR should increase after warmup")


class TestAgentWithUnifiedBuffer(unittest.TestCase):
    """Test agent with unified replay buffer."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not AGENT_AVAILABLE or not UNIFIED_BUFFER_AVAILABLE:
            self.skipTest("Agent or unified buffer not available")
    
    def test_agent_with_unified_buffer(self):
        """Test agent works with unified replay buffer."""
        # Create unified buffer
        buffer = UnifiedReplayBuffer(capacity=1000, device="cpu", prioritized=False)
        
        # Create agent with this buffer
        config = get_minimal_config()
        agent = DQNAgent(
            input_shape=(1, 14, 7),
            n_actions=7,
            device="cpu",
            config=config,
            memory=buffer
        )
        
        # Test that agent uses the provided buffer
        self.assertIs(agent.memory, buffer)
        
        # Test learning with unified buffer
        fill_replay_buffer(agent, 100)
        loss = agent.learn()
        
        self.assertIsNotNone(loss)
        self.assertIsInstance(loss, (int, float))
    
    def test_agent_with_prioritized_buffer(self):
        """Test agent with prioritized unified buffer."""
        # Create prioritized buffer
        buffer = UnifiedReplayBuffer(capacity=1000, device="cpu", prioritized=True)
        
        config = get_minimal_config()
        agent = DQNAgent(
            input_shape=(1, 14, 7),
            n_actions=7,
            device="cpu",
            config=config,
            memory=buffer
        )
        
        # Test learning with prioritized buffer
        fill_replay_buffer(agent, 100)
        loss = agent.learn()
        
        self.assertIsNotNone(loss)
        self.assertIsInstance(loss, (int, float))


class TestAgentErrorHandling(unittest.TestCase):
    """Test agent error handling and edge cases."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not AGENT_AVAILABLE:
            self.skipTest("Agent module not available")
    
    def test_invalid_state_handling(self):
        """Test agent handles invalid states gracefully."""
        agent = create_test_agent()
        
        # Test with None state
        try:
            action = agent.select_action(None, training=True)
            # Should either handle gracefully or raise appropriate error
            self.assertIsInstance(action, (int, np.integer))
        except (ValueError, AttributeError, TypeError):
            # Acceptable to raise these errors for invalid input
            pass
    
    def test_invalid_action_bounds(self):
        """Test that agent actions are always in valid range."""
        agent = create_test_agent()
        state = create_test_state(device=agent.device)
        
        # Test many action selections
        for _ in range(100):
            action = agent.select_action(state, training=True)
            self.assertGreaterEqual(action, 0)
            self.assertLess(action, agent.n_actions)
    
    def test_empty_memory_handling(self):
        """Test agent behavior with empty memory."""
        agent = create_test_agent()
        
        # Try learning with empty memory
        loss = agent.learn()
        self.assertIsNone(loss, "Learning with empty memory should return None")
    
    def test_device_mismatch_handling(self):
        """Test agent handles device mismatches."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        agent = create_test_agent(device="cuda")
        
        # Try action selection with CPU state
        cpu_state = create_test_state(device="cpu")
        
        try:
            action = agent.select_action(cpu_state, training=True)
            # Should handle device transfer or raise appropriate error
            self.assertIsInstance(action, (int, np.integer))
        except RuntimeError:
            # Acceptable to have device mismatch errors
            pass


def run_agent_tests():
    """Run all agent tests and return results."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestDQNAgent))
    if UNIFIED_BUFFER_AVAILABLE:
        suite.addTests(loader.loadTestsFromTestCase(TestAgentWithUnifiedBuffer))
    suite.addTests(loader.loadTestsFromTestCase(TestAgentErrorHandling))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == "__main__":
    print("Testing DQN Agent...")
    print("=" * 50)
    
    # Run tests
    result = run_agent_tests()
    
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
            print(f"- {test}")
            print(f"  {traceback.split(chr(10))[-2] if chr(10) in traceback else traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}")
            print(f"  {traceback.split(chr(10))[-2] if chr(10) in traceback else traceback}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\nOverall: {'PASS' if success else 'FAIL'}")
    
    if success:
        print("✅ All agent tests passed! Agent is ready for training.")
    else:
        print("❌ Some agent tests failed. Please fix issues before training.")
    
    exit(0 if success else 1)