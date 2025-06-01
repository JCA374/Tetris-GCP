"""
Test suite for memory management and replay buffers.

Tests memory usage, GPU memory management, and replay buffer functionality
to ensure efficient memory usage during training.
"""
import unittest
import numpy as np
import torch
import gc
import sys
import os
import time

# Add current directory to path to import local modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from unified_replay_buffer import UnifiedReplayBuffer, create_replay_buffer
    UNIFIED_BUFFER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import unified replay buffer: {e}")
    UNIFIED_BUFFER_AVAILABLE = False

try:
    from replay import ReplayBuffer, PrioritizedReplayBuffer
    LEGACY_BUFFER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import legacy replay buffers: {e}")
    LEGACY_BUFFER_AVAILABLE = False

try:
    from agent import DQNAgent
    from minimal_config import get_minimal_config
    AGENT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import agent: {e}")
    AGENT_AVAILABLE = False


def get_gpu_memory_usage():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return 0


def get_cpu_memory_usage():
    """Get current CPU memory usage in MB."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0


def create_test_transition(device="cpu"):
    """Create a test transition for replay buffer."""
    state = torch.rand(4, 14, 7, device=device)
    action = torch.randint(0, 7, (1,), device=device)
    next_state = torch.rand(4, 14, 7, device=device)
    reward = torch.rand(1, device=device)
    done = torch.rand(1, device=device) < 0.1  # 10% chance of done
    
    return state, action, next_state, reward, done


class TestUnifiedReplayBuffer(unittest.TestCase):
    """Test cases for unified replay buffer."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not UNIFIED_BUFFER_AVAILABLE:
            self.skipTest("Unified replay buffer not available")
    
    def test_buffer_initialization(self):
        """Test buffer initializes correctly."""
        buffer = UnifiedReplayBuffer(capacity=1000, device="cpu")
        
        self.assertEqual(buffer.capacity, 1000)
        self.assertEqual(buffer.device, "cpu")
        self.assertEqual(len(buffer), 0)
        self.assertFalse(buffer.prioritized)
    
    def test_prioritized_buffer_initialization(self):
        """Test prioritized buffer initializes correctly."""
        buffer = UnifiedReplayBuffer(
            capacity=1000, 
            device="cpu", 
            prioritized=True,
            alpha=0.7,
            beta=0.5
        )
        
        self.assertTrue(buffer.prioritized)
        self.assertEqual(buffer.alpha, 0.7)
        self.assertEqual(buffer.beta, 0.5)
    
    def test_buffer_push_and_sample(self):
        """Test basic push and sample functionality."""
        buffer = UnifiedReplayBuffer(capacity=100, device="cpu")
        
        # Add some transitions
        for _ in range(50):
            state, action, next_state, reward, done = create_test_transition("cpu")
            buffer.push(state, action, next_state, reward, done)
        
        self.assertEqual(len(buffer), 50)
        
        # Sample batch
        batch = buffer.sample(32)
        self.assertEqual(len(batch), 32)
        
        # Check transition format
        for transition in batch:
            self.assertEqual(len(transition), 6)  # state, action, next_state, reward, done, info
    
    def test_buffer_capacity_overflow(self):
        """Test buffer handles capacity overflow correctly."""
        buffer = UnifiedReplayBuffer(capacity=10, device="cpu")
        
        # Add more transitions than capacity
        for i in range(20):
            state, action, next_state, reward, done = create_test_transition("cpu")
            buffer.push(state, action, next_state, reward, done)
        
        # Should not exceed capacity
        self.assertEqual(len(buffer), 10)
    
    def test_sample_tensors(self):
        """Test sample_tensors method."""
        buffer = UnifiedReplayBuffer(capacity=100, device="cpu")
        
        # Fill buffer
        for _ in range(50):
            state, action, next_state, reward, done = create_test_transition("cpu")
            buffer.push(state, action, next_state, reward, done)
        
        # Sample as tensors
        result = buffer.sample_tensors(16)
        
        # Standard buffer should return 6 tensors
        self.assertEqual(len(result), 6)
        states, actions, next_states, rewards, dones, non_final_mask = result
        
        # Check tensor shapes
        self.assertEqual(states.shape, (16, 4, 14, 7))
        self.assertEqual(actions.shape, (16,))
        self.assertEqual(rewards.shape, (16,))
        self.assertEqual(dones.shape, (16,))
        self.assertEqual(non_final_mask.shape, (16,))
    
    def test_prioritized_sample_tensors(self):
        """Test sample_tensors method for prioritized buffer."""
        buffer = UnifiedReplayBuffer(capacity=100, device="cpu", prioritized=True)
        
        # Fill buffer
        for _ in range(50):
            state, action, next_state, reward, done = create_test_transition("cpu")
            buffer.push(state, action, next_state, reward, done)
        
        # Sample as tensors
        result = buffer.sample_tensors(16)
        
        # Prioritized buffer should return 8 tensors (includes indices and weights)
        self.assertEqual(len(result), 8)
        states, actions, next_states, rewards, dones, non_final_mask, indices, weights = result
        
        # Check additional outputs for prioritized replay
        self.assertIsInstance(indices, np.ndarray)
        self.assertEqual(indices.shape, (16,))
        self.assertEqual(weights.shape, (16, 1))
    
    def test_update_priorities(self):
        """Test priority update functionality."""
        buffer = UnifiedReplayBuffer(capacity=100, device="cpu", prioritized=True)
        
        # Fill buffer
        for _ in range(50):
            state, action, next_state, reward, done = create_test_transition("cpu")
            buffer.push(state, action, next_state, reward, done)
        
        # Sample and update priorities
        transitions, indices, weights = buffer.sample(16)
        new_priorities = np.random.rand(16)
        
        # Should not raise error
        buffer.update_priorities(indices, new_priorities)
    
    def test_gpu_buffer(self):
        """Test GPU buffer functionality."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        buffer = UnifiedReplayBuffer(capacity=100, device="cuda")
        
        # Add GPU transitions
        for _ in range(20):
            state, action, next_state, reward, done = create_test_transition("cuda")
            buffer.push(state, action, next_state, reward, done)
        
        # Sample should return GPU tensors
        states, actions, _, _, _, _ = buffer.sample_tensors(8)
        
        self.assertEqual(states.device.type, "cuda")
        self.assertEqual(actions.device.type, "cuda")


class TestMemoryManagement(unittest.TestCase):
    """Test memory management and efficiency."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not UNIFIED_BUFFER_AVAILABLE:
            self.skipTest("Unified replay buffer not available")
    
    def test_cpu_memory_growth(self):
        """Test that CPU memory doesn't grow excessively."""
        initial_memory = get_cpu_memory_usage()
        
        # Create and fill multiple buffers
        for _ in range(5):
            buffer = UnifiedReplayBuffer(capacity=1000, device="cpu")
            
            for _ in range(1000):
                state, action, next_state, reward, done = create_test_transition("cpu")
                buffer.push(state, action, next_state, reward, done)
            
            # Sample to trigger memory usage
            for _ in range(10):
                buffer.sample_tensors(32)
            
            del buffer
            gc.collect()
        
        final_memory = get_cpu_memory_usage()
        memory_growth = final_memory - initial_memory
        
        # Memory growth should be reasonable (less than 500MB)
        self.assertLess(memory_growth, 500, 
                       f"Memory grew by {memory_growth:.1f}MB, which may indicate a memory leak")
    
    def test_gpu_memory_cleanup(self):
        """Test GPU memory is properly released."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        initial_memory = get_gpu_memory_usage()
        
        # Create and destroy multiple GPU buffers
        for _ in range(3):
            buffer = UnifiedReplayBuffer(capacity=1000, device="cuda")
            
            for _ in range(500):
                state, action, next_state, reward, done = create_test_transition("cuda")
                buffer.push(state, action, next_state, reward, done)
            
            # Sample to allocate more memory
            for _ in range(5):
                buffer.sample_tensors(64)
            
            del buffer
            torch.cuda.empty_cache()
        
        final_memory = get_gpu_memory_usage()
        memory_growth = final_memory - initial_memory
        
        # GPU memory should not grow significantly
        self.assertLess(memory_growth, 100, 
                       f"GPU memory grew by {memory_growth:.1f}MB, indicating potential memory leak")
    
    def test_buffer_memory_efficiency(self):
        """Test memory efficiency of different buffer types."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        # Test different buffer configurations
        configs = [
            {"prioritized": False, "device": "cpu"},
            {"prioritized": True, "device": "cpu"},
            {"prioritized": False, "device": "cuda"},
            {"prioritized": True, "device": "cuda"},
        ]
        
        for config in configs:
            initial_memory = get_gpu_memory_usage() if config["device"] == "cuda" else get_cpu_memory_usage()
            
            buffer = UnifiedReplayBuffer(capacity=1000, **config)
            
            # Fill buffer
            for _ in range(1000):
                state, action, next_state, reward, done = create_test_transition(config["device"])
                buffer.push(state, action, next_state, reward, done)
            
            # Check memory usage is reasonable
            current_memory = get_gpu_memory_usage() if config["device"] == "cuda" else get_cpu_memory_usage()
            memory_used = current_memory - initial_memory
            
            # Should use less than 200MB for 1000 transitions
            self.assertLess(memory_used, 200, 
                           f"Buffer config {config} used {memory_used:.1f}MB, which seems excessive")
            
            del buffer
            if config["device"] == "cuda":
                torch.cuda.empty_cache()


class TestReplayBufferComparison(unittest.TestCase):
    """Compare unified buffer with legacy buffers."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not UNIFIED_BUFFER_AVAILABLE or not LEGACY_BUFFER_AVAILABLE:
            self.skipTest("Both buffer types not available")
    
    def test_unified_vs_legacy_basic(self):
        """Compare unified buffer with legacy ReplayBuffer."""
        # Create both buffers
        unified = UnifiedReplayBuffer(capacity=100, device="cpu", prioritized=False)
        legacy = ReplayBuffer(capacity=100)
        
        # Add same transitions to both
        transitions = []
        for _ in range(50):
            state, action, next_state, reward, done = create_test_transition("cpu")
            # Convert to numpy for legacy buffer
            state_np = state.cpu().numpy()
            action_np = action.cpu().item()
            next_state_np = next_state.cpu().numpy()
            reward_np = reward.cpu().item()
            done_np = done.cpu().item()
            
            unified.push(state, action, next_state, reward, done)
            legacy.push(state_np, action_np, next_state_np, reward_np, done_np)
            
            transitions.append((state_np, action_np, next_state_np, reward_np, done_np))
        
        # Both should have same length
        self.assertEqual(len(unified), len(legacy))
        
        # Both should be able to sample
        unified_batch = unified.sample(16)
        legacy_batch = legacy.sample(16)
        
        self.assertEqual(len(unified_batch), len(legacy_batch))
    
    def test_unified_vs_legacy_prioritized(self):
        """Compare unified prioritized buffer with legacy PrioritizedReplayBuffer."""
        # Create both buffers
        unified = UnifiedReplayBuffer(
            capacity=100, device="cpu", prioritized=True, alpha=0.6, beta=0.4
        )
        legacy = PrioritizedReplayBuffer(capacity=100, alpha=0.6, beta=0.4)
        
        # Add transitions
        for _ in range(50):
            state, action, next_state, reward, done = create_test_transition("cpu")
            # Convert to numpy for legacy buffer
            state_np = state.cpu().numpy()
            action_np = action.cpu().item()
            next_state_np = next_state.cpu().numpy()
            reward_np = reward.cpu().item()
            done_np = done.cpu().item()
            
            unified.push(state, action, next_state, reward, done)
            legacy.push(state_np, action_np, next_state_np, reward_np, done_np)
        
        # Both should have same length
        self.assertEqual(len(unified), len(legacy))
        
        # Both should return priorities and weights
        unified_result = unified.sample(16)
        legacy_result = legacy.sample(16)
        
        # Both should return 3 items (transitions, indices, weights)
        self.assertEqual(len(unified_result), 3)
        self.assertEqual(len(legacy_result), 3)


class TestAgentMemoryIntegration(unittest.TestCase):
    """Test memory management in full agent training scenario."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not AGENT_AVAILABLE or not UNIFIED_BUFFER_AVAILABLE:
            self.skipTest("Agent or buffer not available")
    
    def test_agent_memory_usage(self):
        """Test memory usage during agent training."""
        initial_memory = get_cpu_memory_usage()
        
        # Create agent with unified buffer
        config = get_minimal_config()
        config["replay_capacity"] = 1000
        config["batch_size"] = 32
        
        buffer = UnifiedReplayBuffer(capacity=1000, device="cpu")
        agent = DQNAgent(
            input_shape=(1, 14, 7),
            n_actions=7,
            device="cpu",
            config=config,
            memory=buffer
        )
        
        # Simulate training
        for episode in range(10):
            for step in range(100):
                state = torch.rand(1, 14, 7)
                action = agent.select_action(state, training=True)
                next_state = torch.rand(1, 14, 7)
                reward = np.random.randn()
                done = step == 99  # End episode at step 99
                
                agent.memory.push(state, action, next_state, reward, done)
                
                # Learn every few steps
                if len(agent.memory) >= agent.batch_size and step % 4 == 0:
                    agent.learn()
        
        final_memory = get_cpu_memory_usage()
        memory_growth = final_memory - initial_memory
        
        # Memory growth should be reasonable
        self.assertLess(memory_growth, 300, 
                       f"Agent memory grew by {memory_growth:.1f}MB during training")
    
    def test_gpu_agent_memory_usage(self):
        """Test GPU memory usage during agent training."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        initial_memory = get_gpu_memory_usage()
        
        # Create GPU agent
        config = get_minimal_config()
        config["replay_capacity"] = 500  # Smaller for GPU test
        config["batch_size"] = 32
        
        buffer = UnifiedReplayBuffer(capacity=500, device="cuda")
        agent = DQNAgent(
            input_shape=(1, 14, 7),
            n_actions=7,
            device="cuda",
            config=config,
            memory=buffer
        )
        
        # Simulate training
        for episode in range(5):
            for step in range(50):
                state = torch.rand(1, 14, 7, device="cuda")
                action = agent.select_action(state, training=True)
                next_state = torch.rand(1, 14, 7, device="cuda")
                reward = np.random.randn()
                done = step == 49
                
                agent.memory.push(state, action, next_state, reward, done)
                
                if len(agent.memory) >= agent.batch_size and step % 4 == 0:
                    agent.learn()
        
        final_memory = get_gpu_memory_usage()
        memory_growth = final_memory - initial_memory
        
        # GPU memory growth should be reasonable
        self.assertLess(memory_growth, 500, 
                       f"GPU memory grew by {memory_growth:.1f}MB during training")


class TestBufferFactoryFunction(unittest.TestCase):
    """Test buffer factory functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not UNIFIED_BUFFER_AVAILABLE:
            self.skipTest("Unified buffer not available")
    
    def test_create_standard_buffer(self):
        """Test creating standard buffer via factory."""
        buffer = create_replay_buffer(
            capacity=1000,
            device="cpu",
            buffer_type="standard"
        )
        
        self.assertIsInstance(buffer, UnifiedReplayBuffer)
        self.assertFalse(buffer.prioritized)
        self.assertEqual(buffer.device, "cpu")
    
    def test_create_prioritized_buffer(self):
        """Test creating prioritized buffer via factory."""
        buffer = create_replay_buffer(
            capacity=1000,
            device="cpu",
            buffer_type="prioritized"
        )
        
        self.assertIsInstance(buffer, UnifiedReplayBuffer)
        self.assertTrue(buffer.prioritized)
    
    def test_create_gpu_buffer(self):
        """Test creating GPU buffer via factory."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        buffer = create_replay_buffer(
            capacity=1000,
            buffer_type="gpu"
        )
        
        self.assertIsInstance(buffer, UnifiedReplayBuffer)
        self.assertEqual(buffer.device, "cuda")
    
    def test_invalid_buffer_type(self):
        """Test invalid buffer type raises error."""
        with self.assertRaises(ValueError):
            create_replay_buffer(
                capacity=1000,
                buffer_type="invalid_type"
            )


def run_memory_tests():
    """Run all memory tests and return results."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    if UNIFIED_BUFFER_AVAILABLE:
        suite.addTests(loader.loadTestsFromTestCase(TestUnifiedReplayBuffer))
        suite.addTests(loader.loadTestsFromTestCase(TestMemoryManagement))
        suite.addTests(loader.loadTestsFromTestCase(TestBufferFactoryFunction))
    
    if UNIFIED_BUFFER_AVAILABLE and LEGACY_BUFFER_AVAILABLE:
        suite.addTests(loader.loadTestsFromTestCase(TestReplayBufferComparison))
    
    if UNIFIED_BUFFER_AVAILABLE and AGENT_AVAILABLE:
        suite.addTests(loader.loadTestsFromTestCase(TestAgentMemoryIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == "__main__":
    print("Testing Memory Management...")
    print("=" * 50)
    
    # Show initial memory usage
    initial_cpu = get_cpu_memory_usage()
    initial_gpu = get_gpu_memory_usage()
    print(f"Initial Memory - CPU: {initial_cpu:.1f}MB, GPU: {initial_gpu:.1f}MB")
    
    # Run tests
    result = run_memory_tests()
    
    # Show final memory usage
    final_cpu = get_cpu_memory_usage()
    final_gpu = get_gpu_memory_usage()
    print(f"Final Memory - CPU: {final_cpu:.1f}MB, GPU: {final_gpu:.1f}MB")
    print(f"Memory Change - CPU: {final_cpu - initial_cpu:+.1f}MB, GPU: {final_gpu - initial_gpu:+.1f}MB")
    
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
        print("✅ All memory tests passed! Memory management is working correctly.")
    else:
        print("❌ Some memory tests failed. Please fix memory issues before training.")
    
    exit(0 if success else 1)