import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import gc
import os
import time
from collections import deque
from model import DQN, DuelDQN
from replay import ReplayBuffer, PrioritizedReplayBuffer

import logging
logger = logging.getLogger('tetris_training')

def get_process_memory():
    """Get the memory usage of the current process in MB."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except (ImportError, AttributeError):
        return 0  # Return 0 if psutil is not available

class DQNAgent:
    """Agent implementing Deep Q-Learning algorithm with various enhancements."""
    
    def __init__(self, input_shape, n_actions, device="cpu", config=None, memory=None, policy_net=None):
        """
        Initialize the DQN agent.
        
        Args:
            input_shape: Shape of the input state
            n_actions: Number of possible actions
            device: Device to run the model on (cpu or cuda)
            config: Dictionary of configuration parameters
            memory: Optional pre-initialized replay buffer
            policy_net: Optional pre-initialized policy network
        """
        self.device = device
        self.config = config or {}
        
        # Hyperparameters (with sensible defaults)
        self.gamma = self.config.get("gamma", 0.99)
        self.learning_rate = self.config.get("learning_rate", 0.0001)
        self.epsilon = self.config.get("epsilon_start", 1.0)
        self.epsilon_end = self.config.get("epsilon_end", 0.1)
        self.epsilon_decay = self.config.get("epsilon_decay", 0.995)
        self.batch_size = self.config.get("batch_size", 32)
        self.target_update = self.config.get("target_update", 10)
        self.replay_capacity = self.config.get("replay_capacity", 10000)
        
        # Model architecture selection
        if policy_net is not None:
            self.policy_net = policy_net.to(device)
            if isinstance(policy_net, DuelDQN):
                self.target_net = DuelDQN(input_shape, n_actions).to(device)
                print("Using provided Dueling DQN architecture")
            else:
                self.target_net = DQN(input_shape, n_actions).to(device)
                print("Using provided standard DQN architecture")
        else:
            model_type = self.config.get("model_type", "dqn").lower()
            if model_type == "dueldqn":
                print("Using Dueling DQN architecture")
                self.policy_net = DuelDQN(input_shape, n_actions).to(device)
                self.target_net = DuelDQN(input_shape, n_actions).to(device)
            else:  # Default to standard DQN
                print("Using standard DQN architecture")
                self.policy_net = DQN(input_shape, n_actions).to(device)
                self.target_net = DQN(input_shape, n_actions).to(device)
            
        # Initialize target network with policy network weights
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.policy_net.parameters(), 
            lr=self.learning_rate,
            weight_decay=self.config.get("weight_decay", 0)  # L2 regularization
        )
        
        # Learning rate scheduler
        self.scheduler = None
        if self.config.get("use_lr_scheduler", False):
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.get("lr_scheduler_step_size", 1000),
                gamma=self.config.get("lr_scheduler_gamma", 0.5),
                last_epoch=-1  # Add this to ensure proper initialization
            )
        
        # Choose replay buffer type
        if memory is not None:
            self.memory = memory
            print(f"Using provided replay buffer of type {type(memory).__name__}")
        else:
            use_per = self.config.get("use_prioritized_replay", False)
            
            if use_per:
                print("Using Prioritized Experience Replay")
                alpha = self.config.get("per_alpha", 0.6)
                beta = self.config.get("per_beta", 0.4)
                self.memory = PrioritizedReplayBuffer(
                    self.replay_capacity, alpha=alpha, beta=beta
                )
            else:
                self.memory = ReplayBuffer(self.replay_capacity)
        
        # Tracking variables
        self.n_actions = n_actions
        self.steps_done = 0
        self.training_steps = 0
        self.episode_count = 0
        
        # Metrics tracking
        self.loss_history = deque(maxlen=100)
        self.q_value_history = deque(maxlen=100)
        self.reward_history = deque(maxlen=100)
        
        # Double DQN flag
        self.use_double_dqn = self.config.get("use_double_dqn", False)
        if self.use_double_dqn:
            print("Using Double DQN algorithm")
            
        # Print memory usage if debugging
        #if self.config.get("debug", False):
            # JCA print(f"Initial agent memory usage: {get_process_memory():.1f} MB")

    def update_epsilon(self):
        """Update exploration rate with decay."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def store_transition(self, state, action, next_state, reward, done, info=None):
        """
        Store a transition in the replay buffer.
        
        Args:
            state: Current state observation
            action: Action taken
            next_state: Next state observation
            reward: Reward received
            done: Boolean indicating if the episode ended
            info: Additional information dictionary from the environment
        """
        self.memory.push(state, action, next_state, reward, done, info)

        # Print buffer size every 100 transitions
        if len(self.memory) % 100 == 0:
            print(f"Buffer size: {len(self.memory)}/{self.batch_size} transitions")

    def get_action(self, state, deterministic=False):
        """
        Wrapper for select_action to match the test expectations.
        
        Args:
            state: The current state
            deterministic: Whether to use deterministic action selection
            
        Returns:
            Selected action
        """
        return self.select_action(state, training=not deterministic)

    def reset(self):
        """
        Reset the agent's state for a new training run.
        
        This resets exploration parameters and episode count but keeps the model weights.
        """
        self.epsilon = self.config.get("epsilon_start", 1.0)
        self.steps_done = 0
        return self

    def learn(self):
        """Update the network weights using a batch of experiences, with debug logging."""
        # 1) Don’t learn until we have enough samples
        if len(self.memory) < self.batch_size:
            logger.info(f"--- learn(): buffer too small {len(self.memory)}/{self.batch_size} ---")
            return None

        try:
            self.training_steps += 1

            # decide when to log: first 10, then every 10
            do_debug = (self.training_steps <= 10) or (self.training_steps % 10 == 0)
            if do_debug:
                logger.info(f"--- Learn Step {self.training_steps} ---")

            # 2) sample
            if hasattr(self.memory, 'sample') and hasattr(self.memory, 'update_priorities'):
                transitions, indices, is_weights = self.memory.sample(self.batch_size)
                importance_weights = is_weights
            else:
                transitions = self.memory.sample(self.batch_size)
                indices = None
                importance_weights = None

            from replay import Transition
            batch = Transition(*zip(*transitions))

            # 3) build tensors without numpy conversion (handles GPU tensors)
            # states
            state_list = [s for s in batch.state if s is not None]
            if isinstance(state_list[0], torch.Tensor):
                state_batch = torch.stack(state_list).float().to(self.device)
            else:
                state_batch = torch.tensor(
                    state_list, dtype=torch.float32, device=self.device
                )

            # actions
            action_vals = [
                a.item() if isinstance(a, torch.Tensor) else a
                for a in batch.action
            ]
            action_batch = torch.tensor(
                action_vals, dtype=torch.long, device=self.device
            )

            # rewards
            reward_vals = [
                r.item() if isinstance(r, torch.Tensor) else r
                for r in batch.reward
            ]
            reward_batch = torch.tensor(
                reward_vals, dtype=torch.float32, device=self.device
            )

            # Apply global reward scaling if configured
            if "reward_scale" in self.config:
                reward_batch = reward_batch * self.config["reward_scale"]

            if do_debug:
                logger.info(
                    f"Sampled Rewards: min={reward_batch.min():.2f}, "
                    f"max={reward_batch.max():.2f}, mean={reward_batch.mean():.2f}"
                )

            # 4) next-state mask
            non_final_mask = torch.tensor(
                [s is not None for s in batch.next_state],
                device=self.device, dtype=torch.bool
            )
            non_final_next = None
            if non_final_mask.any():
                next_list = [s for s in batch.next_state if s is not None]
                if isinstance(next_list[0], torch.Tensor):
                    non_final_next = torch.stack(next_list).float().to(self.device)
                else:
                    non_final_next = torch.tensor(
                        next_list, dtype=torch.float32, device=self.device
                    )

            # 5) Q(s,a)
            q_sa = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))

            # 6) V(s') (Double-DQN if enabled)
            next_vals = torch.zeros(self.batch_size, device=self.device)
            if non_final_next is not None and non_final_mask.any():
                with torch.no_grad():
                    if self.use_double_dqn:
                        next_actions = self.policy_net(non_final_next).max(1)[1].unsqueeze(1)
                        next_vals[non_final_mask] = (
                            self.target_net(non_final_next)
                            .gather(1, next_actions)
                            .squeeze(1)
                        )
                    else:
                        next_vals[non_final_mask] = (
                            self.target_net(non_final_next)
                            .max(1)[0]
                        )

            # 7) target
            target_q = reward_batch + (self.gamma * next_vals)

            if do_debug:
                logger.info(
                    f"Q(s,a): min={q_sa.min():.2f}, max={q_sa.max():.2f}, mean={q_sa.mean():.2f}"
                )
                logger.info(
                    f"V(s'): min={next_vals.min():.2f}, max={next_vals.max():.2f}, mean={next_vals.mean():.2f}"
                )
                logger.info(
                    f"Expected Q: min={target_q.min():.2f}, max={target_q.max():.2f}, mean={target_q.mean():.2f}"
                )

            # 8) PER priorities
            td_errors = q_sa - target_q.unsqueeze(1)
            if indices is not None:
                prios = td_errors.abs().detach().cpu().numpy().flatten()
                self.memory.update_priorities(indices, prios)

            # 9) Huber loss
            loss_tensor = nn.SmoothL1Loss(reduction='none')(
                q_sa, target_q.unsqueeze(1)
            )
            if importance_weights is not None:
                loss_tensor = loss_tensor * importance_weights
            loss = loss_tensor.mean()

            if do_debug:
                logger.info(f"Loss = {loss.item():.4f}, Avg Q = {q_sa.mean().item():.2f}")

                        # 10) backward + step
            self.optimizer.zero_grad()
            loss.backward()
            # 10b) gradient norm logging & clipping
            if self.config.get("clip_gradients", False):
                total_norm = 0.0
                for p in self.policy_net.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.data.norm(2).item() ** 2
                total_norm = total_norm ** 0.5
                logger.info(f"Grad Norm = {total_norm:.4f}")
                torch.nn.utils.clip_grad_norm_(
                    self.policy_net.parameters(),
                    self.config.get("max_grad_norm", 5.0)
                )
            # 10c) optimizer step
            self.optimizer.step()
            if self.scheduler is not None:
               self.scheduler.step()

            # 11) bookkeeping
            self.loss_history.append(loss.item())
            self.q_value_history.append(q_sa.mean().item())
            if (not self.config.get("use_soft_update", False)
                and self.training_steps % (self.target_update * 10) == 0):
                self.update_target_network()

            return loss.item()

        except Exception:
            logger.exception("Error in learn()")
            return None

    def update_target_network(self):
        """Update the target network with the policy network's weights."""
        # Soft update
        if self.config.get("use_soft_update", False):
            tau = self.config.get("tau", 0.01)
            # Ensure tau is applied exactly as specified
            for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)
        # Hard update
        else:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def get_average_loss(self):
        """Get the average loss over recent training steps."""
        if not self.loss_history:
            return None
        return sum(self.loss_history) / len(self.loss_history)
    
    def get_average_q_value(self):
        """Get the average Q-value over recent steps."""
        if not self.q_value_history:
            return None
        return sum(self.q_value_history) / len(self.q_value_history)
    
    def get_average_reward(self):
        """Get the average reward over recent episodes."""
        if not self.reward_history:
            return None
        return sum(self.reward_history) / len(self.reward_history)
    
    def add_episode_reward(self, reward):
        """Add an episode reward to tracking history."""
        self.reward_history.append(reward)
        self.episode_count += 1

    def save(self, path):
        """Save the model to a file."""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Make sure we can serialize the config
        serializable_config = {}
        for k, v in self.config.items():
            if isinstance(v, (int, float, str, bool, type(None))) or isinstance(v, list) or isinstance(v, dict):
                serializable_config[k] = v
        
        try:
            # For each layer, copy CPU tensor to ensure consistency
            def copy_state_dict(state_dict):
                new_state_dict = {}
                for key, param in state_dict.items():
                    # Clone the tensor and move to CPU to ensure consistent saving/loading
                    new_state_dict[key] = param.cpu().detach().clone()
                return new_state_dict
            
            # Create a copy of the state dictionaries
            policy_state_dict = copy_state_dict(self.policy_net.state_dict())
            target_state_dict = copy_state_dict(self.target_net.state_dict())
            
            torch.save({
                'policy_net': policy_state_dict,
                'target_net': target_state_dict,
                'optimizer': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'steps_done': self.steps_done,
                'training_steps': self.training_steps,
                'episode_count': self.episode_count,
                'config': serializable_config
            }, path)
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False

    def load(self, path):
        """Load the model from a file with input shape compatibility checking."""
        if not os.path.exists(path):
            print(f"Error: Checkpoint file {path} does not exist.")
            return {}
                
        try:
            # Check if we need to force a fresh start due to input shape changes
            force_fresh_start = False
            try:
                # Peek at the checkpoint to check input channels
                checkpoint = torch.load(path, map_location=self.device)
                if 'policy_net' in checkpoint:
                    for key, tensor in checkpoint['policy_net'].items():
                        if 'conv.0.weight' in key:  # First conv layer weights
                            # Check input channels (dimension 1 in conv weights)
                            saved_channels = tensor.shape[1]
                            current_channels = self.policy_net.input_shape[0]
                            if saved_channels != current_channels:
                                print(f"\n!!! WARNING: Input channel mismatch detected !!!")
                                print(f"Saved model expects {saved_channels} channels, but current config uses {current_channels} channels")
                                print("This is likely due to changing enhanced preprocessing setting.")
                                print("Forced fresh start to avoid errors...\n")
                                force_fresh_start = True
                                break
            except Exception as e:
                print(f"Error checking checkpoint: {e}")
                
            if force_fresh_start:
                return {}  # Return empty dict to signal fresh start
            
            # Try to load with safer weights_only option if supported
            try:
                checkpoint = torch.load(path, map_location=self.device, weights_only=True)
            except TypeError:
                # Fall back to standard loading if weights_only not supported
                checkpoint = torch.load(path, map_location=self.device)
            
            # Load network parameters
            policy_state_dict = checkpoint['policy_net']
            target_state_dict = checkpoint['target_net']
            
            # Try to load the model parameters with more robust handling
            try:
                # First try with strict=True
                self.policy_net.load_state_dict(policy_state_dict, strict=True)
                self.target_net.load_state_dict(target_state_dict, strict=True)
            except Exception as e:
                print(f"Warning: Could not load model with strict checking: {e}")
                print("Trying with relaxed parameter matching...")
                
                try:
                    # Try with strict=False
                    self.policy_net.load_state_dict(policy_state_dict, strict=False)
                    self.target_net.load_state_dict(target_state_dict, strict=False)
                    print("Loaded model with relaxed compatibility")
                except Exception as inner_e:
                    print(f"Error: Failed to load model parameters: {inner_e}")
                    print("Creating fresh networks. Training will continue from scratch.")
                    return {}
            
            # Make sure the target network is in eval mode
            self.target_net.eval()
            
            # Load optimizer state with safer error handling
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            except Exception as e:
                print(f"Warning: Could not load optimizer state: {e}")
                print("Creating a fresh optimizer.")
            
            # Load agent state variables
            self.epsilon = checkpoint.get('epsilon', self.epsilon)
            self.steps_done = checkpoint.get('steps_done', 0)
            self.training_steps = checkpoint.get('training_steps', 0)
            self.episode_count = checkpoint.get('episode_count', 0)
            
            # Print loaded model info
            print(f"Loaded model from {path}")
            print(f"Model has been trained for {self.episode_count} episodes and {self.steps_done} steps")
            print(f"Current exploration rate (epsilon): {self.epsilon:.4f}")
            
            # Return the loaded config
            return checkpoint.get('config', {})
        except Exception as e:
            print(f"Error loading model from {path}: {e}")
            return {}

    def rebuild_model_for_new_channels(self, input_shape):
        """
        Rebuild the model with a new input shape (channels).
        This allows changing between enhanced and basic preprocessing.
        
        Args:
            input_shape: New input shape with the new number of channels
        """
        print(f"Rebuilding model for new input shape: {input_shape}")
        
        # Determine model type from current policy network
        if isinstance(self.policy_net, DuelDQN):
            print("Creating new Dueling DQN with updated input channels")
            new_policy_net = DuelDQN(input_shape, self.n_actions).to(self.device)
            new_target_net = DuelDQN(input_shape, self.n_actions).to(self.device)
        else:
            print("Creating new standard DQN with updated input channels")
            new_policy_net = DQN(input_shape, self.n_actions).to(self.device)
            new_target_net = DQN(input_shape, self.n_actions).to(self.device)
        
        # Set the new networks
        self.policy_net = new_policy_net
        self.target_net = new_target_net
        self.target_net.eval()
        
        # Reset optimizer for the new network
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=self.learning_rate,                      # original default
            weight_decay=self.config.get("weight_decay", 0)
        )
        # immediately lower it if desired
        for pg in self.optimizer.param_groups:
            pg['lr'] = 3e-5

        
        print("Model rebuilt successfully with new input channels")

    def select_action(self, state, training=True):
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state: Current state observation
            training: Whether the agent is training or evaluating
            
        Returns:
            Selected action
        """
        # Increment steps counter
        self.steps_done += 1
        
        # Epsilon greedy strategy
        if training and random.random() < self.epsilon:
            # Random action
            return random.randint(0, self.n_actions - 1)
        else:
            # Greedy action based on Q-values
            with torch.no_grad():
                # Handle different input types
                if isinstance(state, np.ndarray):
                    # Convert state to tensor if needed
                    state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                elif isinstance(state, torch.Tensor):
                    # If already a tensor, ensure it's on the right device
                    state_tensor = state.to(self.device)
                    if state_tensor.dim() == 3:  # Add batch dimension if needed
                        state_tensor = state_tensor.unsqueeze(0)
                else:
                    # Unknown state type
                    print(f"Warning: Unexpected state type in select_action: {type(state)}")
                    return random.randint(0, self.n_actions - 1)
                
                try:
                    # Get Q-values from policy network
                    q_values = self.policy_net(state_tensor)
                    
                    # Track average Q-value for monitoring (only during training)
                    if training:
                        self.q_value_history.append(q_values.max().item())
                    
                    # Return action with highest Q-value
                    return q_values.max(1)[1].item()
                except Exception as e:
                    print(f"Error in select_action: {e}")
                    return random.randint(0, self.n_actions - 1)