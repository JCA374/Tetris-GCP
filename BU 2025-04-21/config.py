# config.py
import os
import torch

# Check for available CPU cores
cpu_count = os.cpu_count() or 4
# Leave one core free for system processes
optimal_threads = max(1, cpu_count - 1)

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Optimized configuration with GPU support
CONFIG = {
    # Environment settings
    "use_gym_tetris": False,  # Use SimpleTetrisEnv for better performance

    # State representation
    # Use enhanced preprocessing with piece information
    "use_enhanced_preprocessing": True,
    # Keep original state values (not binary)
    "use_binary_states": False,

    # Model settings - Optimized for GPU
    "model_type": "dueldqn",        # Use Dueling DQN for better performance
    "use_double_dqn": True,         # Keep Double DQN as it improves stability
    # Enable prioritized replay to focus on important experiences
    "use_prioritized_replay": True,

    # Learning parameters - Adjusted for GPU
    "learning_rate": 1e-05,      # Keep learning rate
    "batch_size": 512,            # Larger batch size for GPU
    "gamma": 0.995,                # Discount factor for longer-term planning

    # Evaluation settings
    "eval_safety_limit": 5000,    # Maximum steps per evaluation episode
    "eval_early_stop_no_lines": 1500,  # Stop if no lines cleared for this many steps

    # Exploration settings
    "epsilon_start": 1.0,
    "epsilon_end": 0.01,           # Slightly higher minimum exploration to avoid local optima
    "epsilon_decay": 0.9999   ,      # Slower decay for longer exploration

    # Target network updates
    "use_soft_update": True,     # Use soft updates 
    "tau": 0.005,                  # Soft update parameter (removed duplicate)
    "target_update": 100,          # Update target network every 10 episodes

    # Training parameters
    "num_episodes": 10000,         # Extended training for GPU
    "max_steps_per_episode": 3000,  # Longer episodes to allow more learning
    "eval_frequency": 200,         # Evaluate less frequently
    "eval_episodes": 10,          # More evaluation episodes
    "max_eval_steps": 5000,       # Shorter evaluation episodes
    "checkpoint_frequency": 250,   # Frequent checkpoints due to preemption risk

    # Memory settings
    "replay_capacity": 500000,    # Larger replay buffer for GPU

    # — Prioritized Replay hyperparams —
    "pr_alpha": 0.6,              # how much prioritization is used (0 = no prioritization)
    "pr_beta_start": 0.4,         # initial importance‑sampling weight
    "pr_beta_frames": 200000,     # schedule β to 1.0 over this many frames


    # — Multi‑step returns to speed up credit assignment —
    "n_steps": 3,                 # 3‑step returns in your Bellman backups

    # Reward shaping - UPDATED with more balanced values
    "reward_height_weight": -0.5,        # Penalty for board height
    "reward_holes_weight": -1.0,          # Penalty for holes
    "reward_lines_cleared_weight": 1000.0,  # Reward for line clearing
    "reward_bumpiness_weight": -0.3,     # Penalty for uneven surface
    "reward_flat_bonus": 2.0,             # Bonus for flat surfaces
    "reward_survival": 0.5,             # Small bonus just for surviving
    "reward_potential_lines": 5.0,       # Reward for potential lines
    "reward_game_over_penalty": -10.0,    # Penalty for game over
    "reward_tetris_bonus": 2000.0,        # Bonus for clearing 4 lines at once
    "reward_combo_multiplier": 1.25,      # Multiplier for consecutive line clears

    # Curriculum learning
    "use_curriculum_learning": True,      # Enable curriculum learning

    # Reward scale for stability
    "reward_scale": 1.0,                  # Keep reward scale low for stability

    # Training stability
    "clip_gradients": True,
    "max_grad_norm": 1.0,
    "use_lr_scheduler": True,            # Disable LR scheduler for simplicity
    # — Learning‑rate annealing to fine‑tune over 10k episodes —
    "lr_scheduler": {
        "type": "CosineAnnealingLR",
        "T_max": 10000
    },

    # Environment settings
    "episode_timeout": 9000,              # Timeout for stuck episodes

    # Directory settings
    "device": device,                     # Use CUDA if available
    "checkpoint_dir": "checkpoints",
    "model_dir": "models",
    "log_dir": "logs",

    # CPU optimization settings (disabled for GPU)
    "optimize_cpu": False,
    "num_threads": optimal_threads,
    "gc_frequency": 20,
    "pre_allocate_tensors": True,
    "update_frequency": 1,                # Update weights every step
    # Set to True to enable detailed reward component tracking
    "debug": False
}

# Print device information
print(f"Using device: {CONFIG['device']}")
if CONFIG['device'] == 'cuda':
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(
        f"CUDA memory allocated: {torch.cuda.memory_allocated(0) / 1024 / 1024:.2f} MB")
    print(
        f"CUDA memory reserved: {torch.cuda.memory_reserved(0) / 1024 / 1024:.2f} MB")


def get_curriculum_config(config, episode, total_episodes=None):
    """
    Get curriculum-adjusted configuration based on training progress.
    Gradually increases difficulty to guide learning.
    
    Args:
        config: Base configuration
        episode: Current episode number
        total_episodes: Total episodes (defaults to config value)
        
    Returns:
        Adjusted configuration dictionary
    """
    if not config.get("use_curriculum_learning", True):
        return config

    if total_episodes is None:
        total_episodes = config.get("num_episodes", 5000)

    # Make a copy to avoid modifying the original
    adjusted_config = config.copy()

    # Calculate progress (0 to 1)
    progress = episode / total_episodes

    # MODIFICATION: Modified curriculum phases with better hole penalties from the start
    if progress < 0.33:
        # Phase 1: Focus on line clearing but with reasonable hole penalties
        adjusted_config["reward_height_weight"] = -0.5  # Increased from -0.001
        adjusted_config["reward_holes_weight"] = -2.0   # Increased from -0.005
        adjusted_config["reward_bumpiness_weight"] = -0.5  # Increased from 0.0

        # Keep line clear reward significant
        adjusted_config["reward_lines_cleared_weight"] = 5000.0
        adjusted_config["reward_tetris_bonus"] = 10000.0

        # Increase potential lines reward to encourage planning
        adjusted_config["reward_potential_lines"] = 5.0  # Increased from 2.0

        # Keep game over penalty significant
        adjusted_config["reward_game_over_penalty"] = -10.0  # Increased from -3.0
        adjusted_config["reward_survival"] = 0.01  # Increased slightly

    elif progress < 0.66:
        # Phase 2: Slightly stronger penalties
        adjusted_config["reward_height_weight"] = -5.0  # Increased from -0.01
        adjusted_config["reward_holes_weight"] = -3.0   # Increased from -0.05
        adjusted_config["reward_bumpiness_weight"] = -1.0  # Increased from -0.005
        adjusted_config["reward_lines_cleared_weight"] = 1500.0
        adjusted_config["reward_potential_lines"] = 8.0  # Increased from 5.0
        adjusted_config["reward_tetris_bonus"] = 3000.0
        adjusted_config["reward_game_over_penalty"] = -15.0

    else:
        # Phase 3: Final balanced weights
        adjusted_config["reward_height_weight"] = -2.0  # Increased from -0.03
        adjusted_config["reward_holes_weight"] = -5.0   # Increased from -0.1
        adjusted_config["reward_bumpiness_weight"] = -2.0  # Increased from -0.01
        adjusted_config["reward_lines_cleared_weight"] = 2000.0
        adjusted_config["reward_potential_lines"] = 10.0
        adjusted_config["reward_tetris_bonus"] = 3000.0
        adjusted_config["reward_game_over_penalty"] = -20.0

    return adjusted_config
