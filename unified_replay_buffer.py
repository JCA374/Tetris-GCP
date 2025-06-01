import torch
import numpy as np
import random
from collections import namedtuple
from typing import Union, Optional, Tuple, List
import warnings

# Define a transition tuple for the replay buffer
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done', 'info'))


class UnifiedReplayBuffer:
    """
    Unified replay buffer that consolidates all replay buffer variants.
    
    Features:
    - Standard uniform sampling or prioritized sampling
    - GPU acceleration with proper memory management
    - Efficient tensor operations
    - Automatic cleanup on deletion
    - Configurable memory efficiency
    """
    
    def __init__(
        self,
        capacity: int,
        device: str = "cuda",
        prioritized: bool = False,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_annealing: float = 0.001,
        epsilon: float = 1e-5
    ):
        """
        Initialize the unified replay buffer.
        
        Args:
            capacity: Maximum size of the buffer
            device: Device to store tensors on ('cuda' or 'cpu')
            prioritized: Whether to use prioritized replay
            alpha: How much prioritization to use (0 = uniform, 1 = full prioritization)
            beta: Importance sampling correction factor (0 = no correction, 1 = full)
            beta_annealing: Rate at which beta increases towards 1
            epsilon: Small constant to add to priorities to ensure non-zero probability
        """
        self.capacity = capacity
        self.device = device
        self.prioritized = prioritized
        self.memory = []
        self.position = 0
        
        # Prioritized replay parameters
        if prioritized:
            self.alpha = alpha
            self.beta = beta
            self.beta_annealing = beta_annealing
            self.epsilon = epsilon
            self.priorities = np.zeros((capacity,), dtype=np.float32)
            self.max_priority = 1.0
        
    def push(
        self,
        state: Union[np.ndarray, torch.Tensor],
        action: Union[int, torch.Tensor],
        next_state: Optional[Union[np.ndarray, torch.Tensor]],
        reward: Union[float, torch.Tensor],
        done: Union[bool, torch.Tensor],
        info: Optional[dict] = None
    ):
        """
        Add a new transition to the buffer.
        
        Args:
            state: Current state observation
            action: Action taken
            next_state: Next state observation (can be None for terminal states)
            reward: Reward received
            done: Boolean indicating if the episode ended
            info: Additional information dictionary from the environment
        """
        # Convert inputs to appropriate tensor format on target device
        state = self._to_tensor(state)
        action = self._to_tensor(action, dtype=torch.long)
        next_state = self._to_tensor(next_state) if next_state is not None else None
        reward = self._to_tensor(reward, dtype=torch.float32)
        done = self._to_tensor(done, dtype=torch.bool)
        
        # Create transition
        transition = Transition(state, action, next_state, reward, done, info)
        
        # Add to memory
        if len(self.memory) < self.capacity:
            self.memory.append(transition)
        else:
            self.memory[self.position] = transition
        
        # Set priority for prioritized replay
        if self.prioritized:
            self.priorities[self.position] = self.max_priority
        
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Union[List[Transition], Tuple[List[Transition], np.ndarray, torch.Tensor]]:
        """
        Sample a batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            For standard replay: List of transitions
            For prioritized replay: Tuple of (transitions, indices, importance weights)
        """
        buffer_size = len(self.memory)
        
        if buffer_size < batch_size:
            raise ValueError(f"Not enough samples in buffer ({buffer_size}) to sample batch of size {batch_size}")
        
        if self.prioritized:
            return self._sample_prioritized(batch_size)
        else:
            return self._sample_uniform(batch_size)
    
    def sample_tensors(self, batch_size: int) -> Union[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ..., np.ndarray, torch.Tensor]]:
        """
        Sample a batch and return as tensors ready for network processing.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            For standard replay: (states, actions, next_states, rewards, dones, non_final_mask)
            For prioritized replay: (states, actions, next_states, rewards, dones, non_final_mask, indices, weights)
        """
        if self.prioritized:
            transitions, indices, weights = self.sample(batch_size)
        else:
            transitions = self.sample(batch_size)
            indices, weights = None, None
        
        # Convert to tensor batches
        batch = Transition(*zip(*transitions))
        
        state_batch = torch.stack([s for s in batch.state])
        action_batch = torch.cat([a for a in batch.action]).long()
        reward_batch = torch.cat([r for r in batch.reward])
        done_batch = torch.cat([d for d in batch.done])
        
        # Handle non-final next states efficiently
        non_final_mask = ~done_batch
        non_final_next_states = None
        
        if non_final_mask.any():
            non_final_states_list = [s for s, d in zip(batch.next_state, batch.done) if not d.item()]
            if non_final_states_list:
                non_final_next_states = torch.stack(non_final_states_list)
        
        if self.prioritized:
            return state_batch, action_batch, non_final_next_states, reward_batch, done_batch, non_final_mask, indices, weights
        else:
            return state_batch, action_batch, non_final_next_states, reward_batch, done_batch, non_final_mask
    
    def update_priorities(self, indices: np.ndarray, priorities: Union[np.ndarray, torch.Tensor]):
        """
        Update priorities for sampled transitions (only for prioritized replay).
        
        Args:
            indices: Indices of transitions to update
            priorities: New priority values
        """
        if not self.prioritized:
            warnings.warn("update_priorities called on non-prioritized buffer", UserWarning)
            return
        
        # Convert tensor to numpy if needed
        if isinstance(priorities, torch.Tensor):
            priorities = priorities.detach().cpu().numpy()
        
        for idx, priority in zip(indices, priorities):
            # Ensure priority is a scalar
            if isinstance(priority, np.ndarray):
                if priority.size == 1:
                    priority = float(priority.item())
                else:
                    priority = float(np.mean(priority))
            else:
                priority = float(priority)
            
            # Add epsilon and clip to reasonable range
            priority = np.clip(priority + self.epsilon, 1e-5, 100.0)
            
            # Update priority
            if idx < len(self.priorities):
                self.priorities[idx] = priority
                self.max_priority = max(self.max_priority, priority)
    
    def _to_tensor(self, data: Union[np.ndarray, torch.Tensor, float, int, bool], dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Convert input data to tensor on the target device."""
        if data is None:
            return None
        
        if isinstance(data, np.ndarray):
            tensor = torch.from_numpy(data)
        elif isinstance(data, torch.Tensor):
            tensor = data
        else:
            tensor = torch.tensor([data])
        
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
        
        # Move to target device if not already there
        if tensor.device.type != self.device:
            tensor = tensor.to(self.device)
        
        return tensor
    
    def _sample_uniform(self, batch_size: int) -> List[Transition]:
        """Sample uniformly from the buffer."""
        indices = random.sample(range(len(self.memory)), batch_size)
        return [self.memory[idx] for idx in indices]
    
    def _sample_prioritized(self, batch_size: int) -> Tuple[List[Transition], np.ndarray, torch.Tensor]:
        """Sample based on priorities."""
        buffer_size = len(self.memory)
        
        try:
            # Get priorities for current buffer size
            if buffer_size == self.capacity:
                priorities = self.priorities
            else:
                priorities = self.priorities[:buffer_size]
            
            # Convert priorities to probabilities
            probabilities = priorities ** self.alpha
            probabilities_sum = np.sum(probabilities)
            
            # Handle numerical issues
            if probabilities_sum <= 0:
                warnings.warn("Zero or negative sum of priorities detected. Using uniform sampling.")
                probabilities = np.ones_like(probabilities) / len(probabilities)
            else:
                probabilities = probabilities / probabilities_sum
            
            # Sample indices
            indices = np.random.choice(buffer_size, batch_size, replace=False, p=probabilities)
            
            # Get sampled transitions
            samples = [self.memory[idx] for idx in indices]
            
            # Calculate importance sampling weights
            weights = (buffer_size * probabilities[indices]) ** (-self.beta)
            
            # Check for numerical issues
            if np.isnan(weights).any() or np.isinf(weights).any():
                warnings.warn("NaN or Inf detected in importance weights. Using ones.")
                weights = np.ones_like(weights)
            else:
                # Normalize weights
                max_weight = np.max(weights)
                if max_weight > 0:
                    weights = weights / max_weight
            
            # Anneal beta towards 1
            self.beta = min(1.0, self.beta + self.beta_annealing)
            
            # Convert weights to tensor on target device
            weights_tensor = torch.tensor(weights, device=self.device, dtype=torch.float32).unsqueeze(1)
            
            return samples, indices, weights_tensor
            
        except Exception as e:
            warnings.warn(f"Error in prioritized sampling: {e}. Using uniform sampling.")
            # Fallback to uniform sampling
            indices = np.random.choice(buffer_size, batch_size, replace=False)
            samples = [self.memory[idx] for idx in indices]
            weights = torch.ones(batch_size, 1, device=self.device)
            return samples, indices, weights
    
    def __len__(self) -> int:
        """Return the current size of the buffer."""
        return len(self.memory)
    
    def __del__(self):
        """Clean up GPU memory when buffer is deleted."""
        if hasattr(self, 'memory') and self.memory:
            # Clear GPU tensors
            for transition in self.memory:
                if hasattr(transition.state, 'cpu'):
                    del transition.state
                if transition.next_state is not None and hasattr(transition.next_state, 'cpu'):
                    del transition.next_state
                if hasattr(transition.action, 'cpu'):
                    del transition.action
                if hasattr(transition.reward, 'cpu'):
                    del transition.reward
                if hasattr(transition.done, 'cpu'):
                    del transition.done
            
            self.memory.clear()
            
            # Force garbage collection for GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# Factory function for easy creation
def create_replay_buffer(
    capacity: int,
    device: str = "cuda",
    buffer_type: str = "standard",
    **kwargs
) -> UnifiedReplayBuffer:
    """
    Factory function to create replay buffers with common configurations.
    
    Args:
        capacity: Maximum size of the buffer
        device: Device to store tensors on
        buffer_type: Type of buffer ('standard', 'prioritized', 'gpu')
        **kwargs: Additional arguments for the buffer
        
    Returns:
        Configured UnifiedReplayBuffer instance
    """
    if buffer_type == "standard":
        return UnifiedReplayBuffer(capacity=capacity, device=device, prioritized=False, **kwargs)
    elif buffer_type == "prioritized":
        return UnifiedReplayBuffer(capacity=capacity, device=device, prioritized=True, **kwargs)
    elif buffer_type == "gpu":
        return UnifiedReplayBuffer(capacity=capacity, device="cuda", prioritized=False, **kwargs)
    elif buffer_type == "gpu_prioritized":
        return UnifiedReplayBuffer(capacity=capacity, device="cuda", prioritized=True, **kwargs)
    else:
        raise ValueError(f"Unknown buffer_type: {buffer_type}. Choose from 'standard', 'prioritized', 'gpu', 'gpu_prioritized'")