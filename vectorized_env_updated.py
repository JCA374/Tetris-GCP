import numpy as np
import torch
from simple_tetris_env import SimpleTetrisEnv
from train import preprocess_state

class VectorizedTetrisEnv:
    """
    Vectorized environment that runs multiple Tetris environments in parallel.
    This helps to collect experiences more efficiently and keep the GPU fed with data.
    """
    
    def __init__(self, num_envs=8, grid_width=7, grid_height=14, use_enhanced_preprocessing=True, 
                 binary_states=False, device="cuda", env_creator=None):
        """Initialize the vectorized environment."""
        self.num_envs = num_envs
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.device = device
        self.use_enhanced_preprocessing = use_enhanced_preprocessing
        self.binary_states = binary_states
        
        # Create the environments
        if env_creator is not None:
            # Use custom environment creator
            self.envs = [env_creator() for _ in range(num_envs)]
        else:
            # Use default SimpleTetrisEnv
            self.envs = [SimpleTetrisEnv(grid_width=grid_width, grid_height=grid_height) 
                         for _ in range(num_envs)]
        
        # Keep track of episode stats for each environment
        self.episode_rewards = [0.0] * num_envs
        self.episode_lengths = [0] * num_envs
        self.episode_lines_cleared = [0] * num_envs
        
        # Track active environments (not done)
        self.active_envs = [True] * num_envs
        
        # Track cumulative stats across all episodes
        self.total_episodes = 0
        self.total_steps = 0
        self.completed_episodes = 0
        self.all_episode_rewards = []
        self.all_episode_lines = []
    
    def reset(self):
        """Reset all environments and return initial observations."""
        # Reset all environments
        states = [env.reset() for env in self.envs]
        
        # Preprocess states
        processed_states = [
            preprocess_state(
                state, 
                binary=self.binary_states, 
                include_piece_info=self.use_enhanced_preprocessing
            ) for state in states
        ]
        
        # Reset episode stats
        self.episode_rewards = [0.0] * self.num_envs
        self.episode_lengths = [0] * self.num_envs
        self.episode_lines_cleared = [0] * self.num_envs
        self.active_envs = [True] * self.num_envs
        
        return processed_states
    
    def reset_if_done(self):
        """Reset only the environments that are done."""
        # Get current states (reset if done)
        states = []
        for i, env in enumerate(self.envs):
            if not self.active_envs[i]:
                # Environment is done, reset it
                states.append(env.reset())
                # Reset episode stats for this environment
                self.episode_rewards[i] = 0.0
                self.episode_lengths[i] = 0
                self.episode_lines_cleared[i] = 0
                self.active_envs[i] = True
            else:
                # Environment is still active, keep its current state
                states.append(env.reset() if not hasattr(env, '_get_observation') else env._get_observation())
        
        # Preprocess states
        processed_states = [
            preprocess_state(
                state, 
                binary=self.binary_states, 
                include_piece_info=self.use_enhanced_preprocessing
            ) for state in states
        ]
        
        return processed_states
    
    def step(self, actions):
        """Take a step in all environments."""
        results = [env.step(action) for env, action in zip(self.envs, actions)]
        next_states, rewards, dones, infos = zip(*results)
        
        # Track episode stats
        for i in range(self.num_envs):
            if self.active_envs[i]:  # Only update for active environments
                self.episode_rewards[i] += rewards[i]
                self.episode_lengths[i] += 1
                self.episode_lines_cleared[i] += infos[i].get('lines_cleared', 0)
                
                # Check if episode is done
                if dones[i]:
                    self.active_envs[i] = False
                    self.completed_episodes += 1
                    self.all_episode_rewards.append(self.episode_rewards[i])
                    self.all_episode_lines.append(self.episode_lines_cleared[i])
        
        # Preprocess next states
        processed_next_states = [
            preprocess_state(
                next_state, 
                binary=self.binary_states, 
                include_piece_info=self.use_enhanced_preprocessing
            ) if not dones[i] else None
            for i, next_state in enumerate(next_states)
        ]
        
        self.total_steps += self.num_envs
        
        return processed_next_states, rewards, dones, infos
    
    def close(self):
        """Close all environments."""
        for env in self.envs:
            env.close()