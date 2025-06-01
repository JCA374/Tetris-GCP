# Tetris DQN Project Analysis and Debugging Guide

## Project Overview

This is a Deep Q-Learning implementation for playing Tetris with both low-level (individual moves) and high-level (piece placement) action spaces. The project includes GPU optimization, various replay buffer implementations, and extensive configuration options.

## Key Issues Identified

### 1. **Redundant Implementations**
- Multiple replay buffer implementations (standard, prioritized, GPU-based)
- Duplicate preprocessing functions across files
- Inconsistent learning functions (standard vs AMP)

### 2. **Configuration Complexity**
- Learning rate is set in config but manually overridden in some places
- Inconsistent use of enhanced preprocessing
- Curriculum learning parameters may conflict with base parameters

### 3. **Potential Bugs**
- Channel mismatch errors when loading checkpoints
- Memory leaks in GPU replay buffers
- Inconsistent state preprocessing between training and evaluation
- Line clearing detection appears unreliable

### 4. **Performance Issues**
- Excessive logging in training loop
- Inefficient state preprocessing
- Potential GPU memory fragmentation

## Cleanup Tasks

### 1. **Consolidate Replay Buffers**

Create a single, unified replay buffer that can handle all cases:

```python
# Create unified_replay_buffer.py combining the best features
# - GPU support with proper memory management
# - Optional prioritized replay
# - Efficient tensor operations
# - Proper cleanup on deletion
```

### 2. **Standardize Preprocessing**

Move all preprocessing to a single module:

```python
# Create preprocessing.py with:
# - Single preprocess_state function
# - GPU-optimized batch preprocessing
# - Consistent channel ordering
# - Clear documentation of output format
```

### 3. **Fix Learning Rate Management**

```python
# In agent.py, remove manual LR overrides
# Use only config-based LR with proper scheduling
# Add LR warmup if needed for stability
```

### 4. **Simplify Configuration**

```python
# Create a minimal_config.py with only essential parameters
# Remove redundant or conflicting settings
# Add validation to ensure consistency
```

## Test Suite Implementation

### 1. **Environment Tests** (`test_environment.py`)

```python
import unittest
import numpy as np
from simple_tetris_env import SimpleTetrisEnv
from high_level_env import HighLevelTetrisEnv

class TestTetrisEnvironments(unittest.TestCase):
    def test_env_reset(self):
        """Test environment reset returns valid state"""
        env = SimpleTetrisEnv()
        state = env.reset()
        self.assertIsInstance(state, dict)
        self.assertIn('grid', state)
        self.assertEqual(state['grid'].shape, (14, 7))
        
    def test_line_clearing(self):
        """Test line clearing detection and scoring"""
        env = SimpleTetrisEnv()
        env.reset()
        
        # Manually fill bottom row except one cell
        env.grid[13, :-1] = 1
        
        # Place a piece to complete the line
        # Test that line is cleared and reward is given
        
    def test_game_over_detection(self):
        """Test game over is properly detected"""
        env = SimpleTetrisEnv()
        env.reset()
        
        # Fill board to top
        env.grid[0:2, :] = 1
        
        # Spawn new piece should trigger game over
        
    def test_high_level_actions(self):
        """Test high-level action wrapper"""
        env = HighLevelTetrisEnv()
        state = env.reset()
        
        # Verify valid placements are generated
        self.assertGreater(len(env.valid_placements), 0)
        
        # Test action execution
        action = 0
        next_state, reward, done, info = env.step(action)
        self.assertIn('high_level_action', info)
```

### 2. **Preprocessing Tests** (`test_preprocessing.py`)

```python
class TestPreprocessing(unittest.TestCase):
    def test_channel_consistency(self):
        """Test preprocessing produces consistent channels"""
        env = SimpleTetrisEnv()
        state = env.reset()
        
        # Test with different settings
        basic = preprocess_state(state, include_piece_info=False)
        enhanced = preprocess_state(state, include_piece_info=True)
        
        self.assertEqual(basic.shape[0], 1)  # 1 channel
        self.assertEqual(enhanced.shape[0], 4)  # 4 channels
        
    def test_gpu_preprocessing(self):
        """Test GPU preprocessing matches CPU version"""
        # Compare outputs between CPU and GPU preprocessing
        
    def test_batch_preprocessing(self):
        """Test batch preprocessing efficiency"""
        # Test vectorized preprocessing
```

### 3. **Agent Tests** (`test_agent.py`)

```python
class TestDQNAgent(unittest.TestCase):
    def test_learning_step(self):
        """Test single learning step executes without error"""
        agent = create_test_agent()
        
        # Fill replay buffer
        for _ in range(100):
            agent.store_transition(...)
            
        # Test learning
        loss = agent.learn()
        self.assertIsNotNone(loss)
        self.assertIsInstance(loss, float)
        self.assertTrue(0 <= loss <= 100)
        
    def test_action_selection(self):
        """Test epsilon-greedy action selection"""
        agent = create_test_agent()
        
        # Test deterministic selection
        agent.epsilon = 0.0
        actions = [agent.select_action(state) for _ in range(10)]
        self.assertEqual(len(set(actions)), 1)  # All same
        
        # Test exploration
        agent.epsilon = 1.0
        actions = [agent.select_action(state) for _ in range(100)]
        self.assertGreater(len(set(actions)), 1)  # Different actions
        
    def test_model_save_load(self):
        """Test model can be saved and loaded correctly"""
        agent1 = create_test_agent()
        
        # Train briefly
        train_briefly(agent1)
        
        # Save
        agent1.save("test_model.pt")
        
        # Load into new agent
        agent2 = create_test_agent()
        agent2.load("test_model.pt")
        
        # Compare predictions
        state = get_test_state()
        q1 = get_q_values(agent1, state)
        q2 = get_q_values(agent2, state)
        np.testing.assert_array_almost_equal(q1, q2)
```

### 4. **Memory Tests** (`test_memory.py`)

```python
class TestMemoryManagement(unittest.TestCase):
    def test_gpu_memory_cleanup(self):
        """Test GPU memory is properly released"""
        import torch
        
        initial_memory = torch.cuda.memory_allocated()
        
        # Create and destroy multiple agents
        for _ in range(5):
            agent = create_gpu_agent()
            train_briefly(agent)
            del agent
            torch.cuda.empty_cache()
            
        final_memory = torch.cuda.memory_allocated()
        
        # Memory should not grow significantly
        self.assertLess(final_memory - initial_memory, 100 * 1024 * 1024)  # 100MB
        
    def test_replay_buffer_memory(self):
        """Test replay buffer memory usage"""
        # Test that memory usage scales linearly with buffer size
```

### 5. **Integration Tests** (`test_integration.py`)

```python
class TestTrainingIntegration(unittest.TestCase):
    def test_short_training_run(self):
        """Test short training run completes successfully"""
        env = SimpleTetrisEnv()
        agent = create_test_agent()
        
        # Train for 10 episodes
        train(env, agent, num_episodes=10)
        
        # Verify training metrics
        self.assertGreater(agent.episode_count, 0)
        self.assertGreater(len(agent.loss_history), 0)
        
    def test_curriculum_learning(self):
        """Test curriculum learning phases"""
        # Test that reward weights change appropriately
        
    def test_line_clearing_improves(self):
        """Test that agent learns to clear lines"""
        # Train agent and verify line clearing improves over time
```

## Debugging Strategy

### 1. **Start with Minimal Configuration**

```python
MINIMAL_CONFIG = {
    # Core settings only
    "model_type": "dqn",  # Start with basic DQN
    "learning_rate": 1e-4,
    "batch_size": 32,
    "replay_capacity": 10000,
    "gamma": 0.99,
    "epsilon_start": 1.0,
    "epsilon_end": 0.1,
    "epsilon_decay": 0.995,
    "num_episodes": 100,
    
    # Simple reward structure
    "reward_lines_cleared_weight": 1000.0,
    "reward_game_over_penalty": -100.0,
    
    # Disable advanced features initially
    "use_double_dqn": False,
    "use_prioritized_replay": False,
    "use_curriculum_learning": False,
    "use_enhanced_preprocessing": False,
    "use_amp": False,
}
```

### 2. **Progressive Testing**

1. **Phase 1: Basic Functionality**
   - Test with 1 environment, CPU only
   - Verify line clearing works
   - Ensure no crashes for 100 episodes

2. **Phase 2: Add Features**
   - Enable enhanced preprocessing
   - Add Double DQN
   - Test with GPU

3. **Phase 3: Scale Up**
   - Add vectorized environments
   - Enable prioritized replay
   - Add curriculum learning

4. **Phase 4: Optimize**
   - Enable AMP
   - Increase batch size
   - Add multiple environments

### 3. **Monitoring Checklist**

```python
# Add monitoring class
class TrainingMonitor:
    def __init__(self):
        self.metrics = {
            'episodes': [],
            'rewards': [],
            'lines_cleared': [],
            'loss': [],
            'q_values': [],
            'epsilon': [],
            'lr': [],
            'memory_usage': [],
        }
    
    def log_episode(self, episode_data):
        # Log all metrics
        
    def check_health(self):
        # Check for common issues:
        # - Rewards not improving
        # - Loss exploding
        # - Q-values diverging
        # - Memory leaks
        
    def save_checkpoint(self):
        # Save current state for debugging
```

## Training Script Template

```bash
#!/bin/bash
# progressive_training.sh

# Phase 1: Verify basic functionality
echo "Phase 1: Basic Training Test (10 episodes)"
python run_minimal_training.py \
    --episodes 10 \
    --no-gpu \
    --no-curriculum \
    --debug

# Check for errors
if [ $? -ne 0 ]; then
    echo "Phase 1 failed. Check logs."
    exit 1
fi

# Phase 2: Short training run
echo "Phase 2: Short Training (100 episodes)"
python run_minimal_training.py \
    --episodes 100 \
    --save-every 10

# Phase 3: Medium training with GPU
echo "Phase 3: GPU Training (1000 episodes)"
python run_gpu_training.py \
    --episodes 1000 \
    --batch-size 128 \
    --parallel-envs 4

# Phase 4: Full training
echo "Phase 4: Full Training"
python run_gpu_training.py \
    --episodes 10000 \
    --batch-size 512 \
    --parallel-envs 32 \
    --checkpoint-frequency 100
```

## Debugging Specific Issues

### 1. **Line Clearing Not Working**

```python
# Add explicit debugging to SimpleTetrisEnv.step()
def step(self, action):
    # ... existing code ...
    
    # Debug line detection
    for y in range(self.grid_height):
        if np.all(self.grid[y, :] > 0):
            print(f"DEBUG: Full line detected at row {y}")
            print(f"DEBUG: Grid row: {self.grid[y, :]}")
    
    lines_cleared = self._place_piece()
    
    if lines_cleared > 0:
        print(f"DEBUG: Cleared {lines_cleared} lines")
        print(f"DEBUG: Reward calculation: {lines_cleared} * {self.line_clear_weight}")
```

### 2. **GPU Memory Issues**

```python
# Add memory profiling
import torch.cuda

def profile_memory():
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"Reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    
# Call periodically during training
```

### 3. **Learning Not Progressing**

```python
# Add detailed learning diagnostics
def diagnose_learning(agent, num_samples=100):
    # Sample from replay buffer
    transitions = agent.memory.sample(num_samples)
    
    # Analyze rewards
    rewards = [t.reward for t in transitions]
    print(f"Reward stats: min={min(rewards)}, max={max(rewards)}, mean={np.mean(rewards)}")
    
    # Check Q-value predictions
    states = [t.state for t in transitions]
    q_values = [agent.policy_net(s).max().item() for s in states]
    print(f"Q-value stats: min={min(q_values)}, max={max(q_values)}, mean={np.mean(q_values)}")
    
    # Check gradient norms
    # Check loss stability
```

## Final Recommendations

1. **Start Simple**: Begin with the minimal configuration and basic DQN
2. **Test Incrementally**: Add features one at a time
3. **Monitor Everything**: Log all metrics for debugging
4. **Save Frequently**: Checkpoint every N episodes
5. **Verify Basics First**: Ensure line clearing works before optimizing
6. **Use CPU First**: Debug on CPU before moving to GPU
7. **Profile Memory**: Monitor memory usage throughout training
8. **Validate Preprocessing**: Ensure consistent state representation

This systematic approach should help identify and fix issues progressively.