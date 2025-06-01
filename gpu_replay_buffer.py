import torch
import numpy as np
from collections import namedtuple

# Define a transition tuple for the replay buffer
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done', 'info'))

class GPUReplayBuffer:
    """
    Memory-efficient GPU replay buffer for DQN.
    Stores transitions directly on the GPU to eliminate CPU-GPU transfer overhead.
    """
    
    def __init__(self, capacity, device="cuda"):
        self.capacity = capacity
        self.device = device
        self.memory = []
        self.position = 0
        
    def push(self, state, action, next_state, reward, done, info=None):
        """
        Add a new transition to the buffer, converting to tensors on the GPU.
        
        Args:
            state: Current state observation
            action: Action taken
            next_state: Next state observation
            reward: Reward received
            done: Boolean indicating if the episode ended
            info: Additional information dictionary from the environment
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
        
        # Store transition
        transition = Transition(state, action, next_state, reward, done, info)
        
        # Add to memory
        if len(self.memory) < self.capacity:
            self.memory.append(transition)
        else:
            self.memory[self.position] = transition
        
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        """
        Sample a batch of transitions.
        Returns tensors already on the GPU, avoiding transfer overhead.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Batch of transitions as a Transition namedtuple of tensors
        """
        # Get random indices
        indices = torch.randint(0, len(self.memory), (batch_size,), device=self.device)
        
        # Extract transitions
        transitions = [self.memory[idx] for idx in indices.cpu().numpy()]
        
        # Directly return the batch
        return transitions

    def sample_tensors(self, batch_size):
        """
        Sample a batch and return as tensors ready for network processing.
        This is more efficient than sample() because it avoids CPU-GPU transfers.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of (states, actions, next_states, rewards, dones, non_final_mask)
        """
        # Get random indices
        indices = torch.randint(0, len(self.memory), (batch_size,), device="cpu")
        
        # Extract transitions
        transitions = [self.memory[idx] for idx in indices.numpy()]
        batch = Transition(*zip(*transitions))
        
        # Create batch tensors - already on the GPU
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
        
        return state_batch, action_batch, non_final_next_states, reward_batch, done_batch, non_final_mask

    def __len__(self):
        """Return the current size of the buffer."""
        return len(self.memory)


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