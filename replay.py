import random
import numpy as np
import zlib
import pickle
import os
import tempfile
from collections import namedtuple, deque
import torch

# Define a transition tuple for the replay buffer - Added 'info' to the fields
Transition = namedtuple('Transition', 
                        ('state', 'action', 'next_state', 'reward', 'done', 'info'))

class ReplayBuffer:
    """Memory-efficient standard experience replay buffer."""
    
    def __init__(self, capacity):
        self.memory = []
        self.capacity = capacity
        self.position = 0
    
    def push(self, state, action, next_state, reward, done, info=None):
        """Add a new transition to the buffer."""
        # Store the full transition
        transition = Transition(state, action, next_state, reward, done, info)
        
        if len(self.memory) < self.capacity:
            self.memory.append(transition)
        else:
            self.memory[self.position] = transition
        
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        """Sample a batch of transitions."""
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

import torch
import numpy as np
from collections import namedtuple

# Define a transition tuple for the replay buffer
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done', 'info'))

class GPUPrioritizedReplayBuffer:
    """
    GPU-accelerated Prioritized Experience Replay buffer.
    Implements importance sampling with prioritization based on TD-error.
    """
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_annealing=0.001, epsilon=1e-5, device="cuda"):
        """
        Initialize the prioritized replay buffer with GPU support.
        
        Args:
            capacity: Maximum size of the buffer
            alpha: How much prioritization to use (0 = uniform sampling, 1 = full prioritization)
            beta: Importance sampling correction factor (0 = no correction, 1 = full correction)
            beta_annealing: Rate at which beta increases (towards 1)
            epsilon: Small constant to add to priorities to ensure non-zero sampling probability
            device: Device to store tensors on (cuda or cpu)
        """
        self.capacity = capacity
        self.device = device
        self.memory = []
        # Store priorities on CPU - these are accessed less frequently during training
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.alpha = alpha
        self.beta = beta
        self.beta_annealing = beta_annealing
        self.epsilon = epsilon
        self.max_priority = 1.0  # Initial max priority  

    def push(self, state, action, next_state, reward, done, info=None):
        """
        Add a new transition to the buffer with max priority.
        Converts inputs to GPU tensors.
        """
        # Convert numpy arrays to tensors if needed
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float().to(self.device)
        elif isinstance(state, torch.Tensor) and state.device.type != self.device:
            state = state.to(self.device)
            
        # Handle action (can be scalar or tensor)
        if not isinstance(action, torch.Tensor):
            action = torch.tensor([action], device=self.device)
        elif action.device.type != self.device:
            action = action.to(self.device)
            
        # Handle next_state (can be None for terminal states)
        if next_state is not None:
            if isinstance(next_state, np.ndarray):
                next_state = torch.from_numpy(next_state).float().to(self.device)
            elif isinstance(next_state, torch.Tensor) and next_state.device.type != self.device:
                next_state = next_state.to(self.device)
        
        # Handle reward (can be scalar or tensor)
        if not isinstance(reward, torch.Tensor):
            reward = torch.tensor([reward], device=self.device, dtype=torch.float32)
        elif reward.device.type != self.device:
            reward = reward.to(self.device)
            
        # Handle done flag
        if not isinstance(done, torch.Tensor):
            done = torch.tensor([done], device=self.device, dtype=torch.bool)
        elif done.device.type != self.device:
            done = done.to(self.device)
            
        # Create transition
        transition = Transition(state, action, next_state, reward, done, info)
        
        # If buffer not full, add new entry
        if len(self.memory) < self.capacity:
            self.memory.append(transition)
        # Otherwise, replace old entry
        else:
            self.memory[self.position] = transition
        
        # Set priority to max priority for new transitions
        self.priorities[self.position] = self.max_priority
        
        # Update position counter
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        Sample a batch of transitions based on priorities.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of (batch of transitions, indices, importance sampling weights tensor on GPU)
        """
        # Get current buffer size
        buffer_size = len(self.memory)
        
        # Check if we have enough samples
        if buffer_size < batch_size:
            raise ValueError(f"Not enough samples in buffer ({buffer_size}) to sample batch of size {batch_size}")
        
        try:
            # Calculate sampling probabilities based on priorities
            if buffer_size == self.capacity:
                priorities = self.priorities
            else:
                priorities = self.priorities[:buffer_size]
            
            # Convert priorities to probabilities using alpha
            probabilities = priorities ** self.alpha
            probabilities_sum = np.sum(probabilities)
            
            # Check for numerical issues
            if probabilities_sum <= 0:
                print("Warning: Zero or negative sum of priorities detected. Using uniform sampling.")
                probabilities = np.ones_like(probabilities) / len(probabilities)
            else:
                probabilities = probabilities / probabilities_sum
            
            # Sample indices based on probabilities
            indices = np.random.choice(buffer_size, batch_size, replace=False, p=probabilities)
            
            # Get sampled transitions
            samples = [self.memory[idx] for idx in indices]
            
            # Calculate importance sampling weights
            weights = (buffer_size * probabilities[indices]) ** (-self.beta)
            
            # CRITICAL: Check for NaN/Inf values in weights
            if np.isnan(weights).any() or np.isinf(weights).any():
                print("Warning: NaN or Inf detected in importance weights. Using ones.")
                weights = np.ones_like(weights)
            else:
                # Normalize to scale between 0 and 1
                max_weight = np.max(weights)
                if max_weight > 0:
                    weights = weights / max_weight
            
            # Anneal beta towards 1 (full compensation)
            self.beta = min(1.0, self.beta + self.beta_annealing)
            
            # Convert weights to tensor (on GPU)
            weights_tensor = torch.tensor(weights, device=self.device, dtype=torch.float32).unsqueeze(1)
            
            return samples, indices, weights_tensor
            
        except Exception as e:
            print(f"Error in GPUPrioritizedReplayBuffer.sample: {e}. Using uniform sampling.")
            # Fallback to uniform sampling
            indices = np.random.choice(buffer_size, batch_size, replace=False)
            samples = [self.memory[idx] for idx in indices]
            weights = torch.ones(batch_size, 1, device=self.device)
            return samples, indices, weights

    def sample_tensors(self, batch_size):
        """
        Sample a batch and return as tensors ready for network processing.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of (states, actions, next_states, rewards, dones, non_final_mask, indices, weights)
        """
        transitions, indices, weights = self.sample(batch_size)
        batch = Transition(*zip(*transitions))
        
        # Create batch tensors
        state_batch = torch.stack([s for s in batch.state])
        action_batch = torch.cat([a for a in batch.action]).long()
        reward_batch = torch.cat([r for r in batch.reward])
        done_batch = torch.cat([d for d in batch.done])
        
        # Handle non-final next states
        non_final_mask = ~done_batch
        
        # Efficiently handle next_states - only process non-terminals
        non_final_next_states = None
        if non_final_mask.any():
            non_final_states_list = [s for s, d in zip(batch.next_state, batch.done) if not d.item()]
            if non_final_states_list:
                non_final_next_states = torch.stack(non_final_states_list)
        
        return state_batch, action_batch, non_final_next_states, reward_batch, done_batch, non_final_mask, indices, weights

    def update_priorities(self, indices, priorities):
        """
        Update priorities for sampled transitions.
        
        Args:
            indices: Indices of transitions to update
            priorities: New priority values (tensor or numpy array)
        """
        # If priorities is a tensor, convert to numpy
        if isinstance(priorities, torch.Tensor):
            priorities = priorities.detach().cpu().numpy()
        
        for idx, priority in zip(indices, priorities):
            # Add small epsilon to avoid zero probability and ensure it's a scalar
            if isinstance(priority, np.ndarray):
                # Handle arrays of different sizes
                if priority.size == 1:
                    priority = float(priority.item() + self.epsilon)
                else:
                    # If the array has multiple elements, take the mean
                    priority = float(np.mean(priority) + self.epsilon)
            else:
                priority = float(priority + self.epsilon)
            
            # Clip priority to reasonable range to avoid numerical issues
            priority = min(max(priority, 1e-5), 100.0)
            
            # Update priority at the correct index
            if idx < len(self.priorities):
                self.priorities[idx] = priority
                
                # Update max priority
                self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        """Return the current size of the buffer."""
        return len(self.memory)

class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer.
    Implements importance sampling with prioritization based on TD-error.
    """
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_annealing=0.001, epsilon=1e-5):
        """
        Initialize the prioritized replay buffer.
        
        Args:
            capacity: Maximum size of the buffer
            alpha: How much prioritization to use (0 = uniform sampling, 1 = full prioritization)
            beta: Importance sampling correction factor (0 = no correction, 1 = full correction)
            beta_annealing: Rate at which beta increases (towards 1)
            epsilon: Small constant to add to priorities to ensure non-zero sampling probability
        """
        self.capacity = capacity
        self.memory = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.alpha = alpha
        self.beta = beta
        self.beta_annealing = beta_annealing
        self.epsilon = epsilon
        self.max_priority = 1.0  # Initial max priority  

    def push(self, state, action, next_state, reward, done, info=None):
        """
        Add a new transition to the buffer with max priority.
        
        Args:
            state: Current state observation
            action: Action taken
            next_state: Next state observation
            reward: Reward received
            done: Boolean indicating if the episode ended
            info: Additional information dictionary from the environment
        """
        # Create new transition
        transition = Transition(state, action, next_state, reward, done, info)
        
        # If buffer not full, add new entry
        if len(self.memory) < self.capacity:
            self.memory.append(transition)
        # Otherwise, replace old entry
        else:
            self.memory[self.position] = transition
        
        # Set priority to max priority for new transitions
        self.priorities[self.position] = self.max_priority
        
        # Update position counter
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        Sample a batch of transitions based on priorities.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of (batch of transitions, indices, importance sampling weights)
        """
        # Get current buffer size
        buffer_size = len(self.memory)
        
        # Check if we have enough samples
        if buffer_size < batch_size:
            raise ValueError(f"Not enough samples in buffer ({buffer_size}) to sample batch of size {batch_size}")
        
        try:
            # Calculate sampling probabilities based on priorities
            if buffer_size == self.capacity:
                priorities = self.priorities
            else:
                priorities = self.priorities[:buffer_size]
            
            # Convert priorities to probabilities using alpha
            probabilities = priorities ** self.alpha
            probabilities_sum = np.sum(probabilities)
            
            # Check for numerical issues
            if probabilities_sum <= 0:
                print("Warning: Zero or negative sum of priorities detected. Using uniform sampling.")
                probabilities = np.ones_like(probabilities) / len(probabilities)
            else:
                probabilities = probabilities / probabilities_sum
            
            # Sample indices based on probabilities
            indices = np.random.choice(buffer_size, batch_size, replace=False, p=probabilities)
            
            # Get sampled transitions
            samples = [self.memory[idx] for idx in indices]
            
            # Calculate importance sampling weights
            weights = (buffer_size * probabilities[indices]) ** (-self.beta)
            
            # CRITICAL: Check for NaN/Inf values in weights
            if np.isnan(weights).any() or np.isinf(weights).any():
                print("Warning: NaN or Inf detected in importance weights. Using ones.")
                weights = np.ones_like(weights)
            else:
                # Normalize to scale between 0 and 1
                max_weight = np.max(weights)
                if max_weight > 0:
                    weights = weights / max_weight
            
            # Anneal beta towards 1 (full compensation)
            self.beta = min(1.0, self.beta + self.beta_annealing)
            
            return samples, indices, weights
            
        except Exception as e:
            print(f"Error in PrioritizedReplayBuffer.sample: {e}. Using uniform sampling.")
            # Fallback to uniform sampling
            indices = np.random.choice(buffer_size, batch_size, replace=False)
            samples = [self.memory[idx] for idx in indices]
            weights = np.ones(batch_size)
            return samples, indices, weights

    def update_priorities(self, indices, priorities):
        """
        Update priorities for sampled transitions.
        
        Args:
            indices: Indices of transitions to update
            priorities: New priority values
        """
        for idx, priority in zip(indices, priorities):
            # Add small epsilon to avoid zero probability and ensure it's a scalar
            if isinstance(priority, np.ndarray):
                # Handle arrays of different sizes
                if priority.size == 1:
                    priority = float(priority.item() + self.epsilon)
                else:
                    # If the array has multiple elements, take the mean
                    priority = float(np.mean(priority) + self.epsilon)
            else:
                priority = float(priority + self.epsilon)
            
            # Clip priority to reasonable range to avoid numerical issues
            priority = min(max(priority, 1e-5), 100.0)
            
            # Update priority at the correct index
            if idx < len(self.priorities):
                self.priorities[idx] = priority
                
                # Update max priority
                self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        """Return the current size of the buffer."""
        return len(self.memory)

class CompressedReplayBuffer:
    """Memory-optimized replay buffer that compresses state representations."""
    
    def __init__(self, capacity, compression_level=1):
        """
        Initialize the compressed replay buffer.
        
        Args:
            capacity: Maximum size of the buffer
            compression_level: Zlib compression level (0-9), higher is more compression
        """
        self.memory = []
        self.capacity = capacity
        self.position = 0
        self.compression_level = compression_level
    
    def _compress_state(self, state):
        """Compress a state to reduce memory usage."""
        # Convert to numpy array if not already
        if not isinstance(state, np.ndarray):
            state = np.array(state)
        
        # Use lower precision
        if state.dtype == np.float64 or state.dtype == np.float32:
            state = state.astype(np.float16)
        
        # For binary/sparse states, use boolean or uint8
        if np.array_equal(state, state.astype(bool)):
            state = state.astype(bool)
        
        # For higher compression levels, use zlib
        if self.compression_level > 0:
            state_bytes = pickle.dumps(state)
            compressed = zlib.compress(state_bytes, self.compression_level)
            return compressed
        
        return state
    
    def _decompress_state(self, compressed_state):
        """Decompress a state."""
        if self.compression_level > 0 and isinstance(compressed_state, bytes):
            state_bytes = zlib.decompress(compressed_state)
            return pickle.loads(state_bytes)
        return compressed_state
    
    def push(self, state, action, next_state, reward, done, info=None):
        """
        Add a new transition to the buffer with compression.
        
        Args:
            state: Current state observation
            action: Action taken
            next_state: Next state observation
            reward: Reward received
            done: Boolean indicating if the episode ended
            info: Additional information dictionary from the environment
        """
        # Compress states to save memory
        compressed_state = self._compress_state(state)
        compressed_next_state = self._compress_state(next_state) if next_state is not None else None
        
        # Create transition
        transition = Transition(compressed_state, action, compressed_next_state, reward, done, info)
        
        # Add to memory
        if len(self.memory) < self.capacity:
            self.memory.append(transition)
        else:
            self.memory[self.position] = transition
        
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        """
        Sample a batch of transitions and decompress states.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Batch of transitions with decompressed states
        """
        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        
        # Decompress states when sampling
        decompressed_batch = []
        for idx in indices:
            c_state, action, c_next_state, reward, done, info = self.memory[idx]
            state = self._decompress_state(c_state)
            next_state = self._decompress_state(c_next_state) if c_next_state is not None else None
            decompressed_batch.append(Transition(state, action, next_state, reward, done, info))
        
        return decompressed_batch
    
    def __len__(self):
        """Return the current size of the buffer."""
        return len(self.memory)

class MemoryMappedReplayBuffer:
    """
    Replay buffer that uses memory-mapped arrays for state storage.
    This allows handling very large replay buffers without loading everything into RAM.
    """
    
    def __init__(self, capacity, state_shape, temp_dir=None):
        """
        Initialize memory-mapped replay buffer.
        
        Args:
            capacity: Maximum size of the buffer
            state_shape: Shape of state observations
            temp_dir: Directory to store memory-mapped files
        """
        self.capacity = capacity
        self.position = 0
        self.size = 0
        self.state_shape = state_shape
        
        # Create temporary directory for memory-mapped files
        if temp_dir is None:
            self.temp_dir = tempfile.mkdtemp()
        else:
            os.makedirs(temp_dir, exist_ok=True)
            self.temp_dir = temp_dir
        
        # Create memory-mapped arrays
        self.states_filename = os.path.join(self.temp_dir, 'states.dat')
        self.next_states_filename = os.path.join(self.temp_dir, 'next_states.dat')
        
        states_shape = (capacity,) + state_shape
        self.states = np.memmap(self.states_filename, dtype=np.float16, 
                               mode='w+', shape=states_shape)
        self.next_states = np.memmap(self.next_states_filename, dtype=np.float16, 
                                    mode='w+', shape=states_shape)
        
        # These are small enough to keep in memory
        self.actions = np.zeros(capacity, dtype=np.int8)
        self.rewards = np.zeros(capacity, dtype=np.float16)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        self.infos = [None] * capacity  # For storing info dictionaries
    
    def push(self, state, action, next_state, reward, done, info=None):
        """Add a transition to the buffer."""
        # Store state and next_state in memory-mapped arrays
        self.states[self.position] = state.astype(np.float16)
        
        if next_state is not None:
            self.next_states[self.position] = next_state.astype(np.float16)
        
        # Store other elements in memory
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.dones[self.position] = done
        self.infos[self.position] = info
        
        # Update position and size
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size):
        """Sample a batch of transitions."""
        indices = np.random.choice(self.size, batch_size, replace=False)
        
        # Create batch of transitions
        states = self.states[indices].astype(np.float32)
        next_states = self.next_states[indices].astype(np.float32)
        actions = self.actions[indices]
        rewards = self.rewards[indices].astype(np.float32)
        dones = self.dones[indices]
        infos = [self.infos[idx] for idx in indices]
        
        batch = []
        for i in range(batch_size):
            # Don't return next_state if episode ended
            next_state = None if dones[i] else next_states[i]
            batch.append(Transition(states[i], actions[i], next_state, rewards[i], dones[i], infos[i]))
        
        return batch
    
    def __len__(self):
        """Return the current size of the buffer."""
        return self.size
    
    def __del__(self):
        """Clean up memory-mapped files when buffer is deleted."""
        # Close memory-mapped files
        if hasattr(self, 'states'):
            self.states._mmap.close()
        if hasattr(self, 'next_states'):
            self.next_states._mmap.close()
        
        # Remove temp files
        if os.path.exists(self.states_filename):
            os.remove(self.states_filename)
        if os.path.exists(self.next_states_filename):
            os.remove(self.next_states_filename)