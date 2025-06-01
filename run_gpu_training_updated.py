#!/usr/bin/env python
"""
Script to run DQN Tetris training on GPU.
Optimized for GCP GPU VM (NVIDIA T4) with preemption handling.
Updated to support high-level actions.
"""
import os
import sys
import argparse
import torch
import numpy as np
import traceback
import time
import signal
import atexit
import logging

from simple_tetris_env import SimpleTetrisEnv
from agent import DQNAgent
from train import train, preprocess_state
from config import CONFIG
from vectorized_env_updated import VectorizedTetrisEnv

# Default root logger level
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Throttle CPU threads so GPU does most of the work
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# Enable CUDA optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run DQN Tetris training on GPU")
    parser.add_argument("--episodes", type=int, default=CONFIG.get("num_episodes", 10000),
                        help="Number of episodes to train")
    parser.add_argument("--render", action="store_true",
                        help="Render environment during training (not recommended for headless VMs)")
    parser.add_argument("--force-basic-preprocessing", action="store_true",
                        help="Force use of basic state preprocessing (disable enhanced)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from latest checkpoint")
    parser.add_argument("--no-curriculum", action="store_true",
                        help="Disable curriculum learning")
    parser.add_argument("--eval-only", action="store_true",
                        help="Only run evaluation on best model")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug output")
    parser.add_argument("--grid-width", type=int, default=7,
                        help="Width of the Tetris grid")
    parser.add_argument("--grid-height", type=int, default=14,
                        help="Height of the Tetris grid")
    parser.add_argument("--parallel-envs", type=int, default=32,
                        help="Number of parallel environments to use (default=32)")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="Batch size for training (default=512)")
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable Automatic Mixed Precision (AMP)")
    parser.add_argument("--log-file", type=str, default="training.log",
                        help="File to log output to")
    parser.add_argument("--high-level-actions", action="store_true",
                       help="Use high-level actions (piece placements) instead of low-level controls")
    return parser.parse_args()


def setup_logging(log_file, level=logging.INFO):
    """Set up 'tetris_training' logger with console and file handlers."""
    tlog = logging.getLogger('tetris_training')
    tlog.setLevel(level)
    # console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    tlog.addHandler(ch)
    # file handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    tlog.addHandler(fh)
    return tlog


def save_checkpoint_on_exit(agent, path, logger):
    """Save checkpoint when the script exits."""
    logger.info("Saving checkpoint before exit...")
    try:
        agent.save(path)
        logger.info(f"Checkpoint saved to {path}")
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")


def main():
    args = parse_args()

    # configure 'tetris_training' logger after args parsed
    level = logging.INFO if args.debug else logging.WARNING
    logger = setup_logging(args.log_file, level=level)

    # propagate debug flag
    CONFIG["debug"] = args.debug

    # override config with CLI args
    CONFIG["num_episodes"] = args.episodes
    CONFIG["use_curriculum_learning"] = not args.no_curriculum
    CONFIG["batch_size"] = args.batch_size
    CONFIG["use_amp"] = not args.no_amp
    if args.force_basic_preprocessing:
        CONFIG["use_enhanced_preprocessing"] = False

    # log system info
    logger.info("\n===== GPU DQN TETRIS TRAINING =====")
    logger.info(f"Episodes: {CONFIG['num_episodes']}")
    logger.info(f"Curriculum: {CONFIG['use_curriculum_learning']}")
    logger.info(f"Batch size: {CONFIG['batch_size']}")
    logger.info(f"Parallel envs: {args.parallel_envs}")
    logger.info(f"Resume: {args.resume}")
    logger.info(f"Debug: {args.debug}")
    logger.info(f"High-level actions: {args.high_level_actions}")

    # CUDA setup
    if torch.cuda.is_available():
        logger.info(f"CUDA: {torch.cuda.get_device_name(0)} (v{torch.version.cuda})")
        CONFIG["device"] = "cuda"
    else:
        logger.warning("CUDA not available, using CPU")
        CONFIG["device"] = "cpu"
        CONFIG["use_amp"] = False

    # Create environment (with high-level action support)
    if args.high_level_actions:
        logger.info("Using high-level actions for Tetris environment")
        from high_level_env import HighLevelTetrisEnv
        
        if args.parallel_envs > 1:
            # Create vectorized environment with high-level wrappers
            logger.info(f"Creating vectorized envs with high-level actions: {args.parallel_envs}")
            
            # Define environment creator function
            def env_creator():
                return HighLevelTetrisEnv(grid_width=args.grid_width, grid_height=args.grid_height)
            
            env = VectorizedTetrisEnv(
                num_envs=args.parallel_envs,
                grid_width=args.grid_width,
                grid_height=args.grid_height,
                use_enhanced_preprocessing=CONFIG.get("use_enhanced_preprocessing", True),
                binary_states=CONFIG.get("use_binary_states", False),
                device=CONFIG.get("device", "cuda"),
                env_creator=env_creator
            )
        else:
            # Single environment with high-level actions
            env = HighLevelTetrisEnv(grid_width=args.grid_width, grid_height=args.grid_height)
    else:
        # Standard low-level action environment
        logger.info(f"Using standard low-level actions")
        if args.parallel_envs > 1:
            logger.info(f"Creating vectorized envs: {args.parallel_envs}")
            env = VectorizedTetrisEnv(
                num_envs=args.parallel_envs,
                grid_width=args.grid_width,
                grid_height=args.grid_height,
                use_enhanced_preprocessing=CONFIG.get("use_enhanced_preprocessing", True),
                binary_states=CONFIG.get("use_binary_states", False),
                device=CONFIG.get("device", "cuda")
            )
        else:
            env = SimpleTetrisEnv(grid_width=args.grid_width, grid_height=args.grid_height)

    # determine input shape
    states = env.reset()
    if hasattr(env, "num_envs"):
        input_shape = states[0].shape if states else (1, args.grid_height, args.grid_width)
    else:
        input_shape = preprocess_state(
            states,
            binary=CONFIG.get("use_binary_states", False),
            include_piece_info=CONFIG.get("use_enhanced_preprocessing", True)
        ).shape

    # Determine action space size
    if args.high_level_actions:
        if hasattr(env, "num_envs"):
            # Vectorized env - get from first wrapped env
            n_actions = env.envs[0].action_space.n
        else:
            n_actions = env.action_space.n
        logger.info(f"Action space size (high-level): {n_actions}")
    else:
        n_actions = 6  # Standard Tetris actions
        logger.info(f"Action space size (low-level): {n_actions}")

    # create replay buffer & agent
    from gpu_replay_buffer import GPUReplayBuffer
    memory = GPUReplayBuffer(
        capacity=CONFIG.get("replay_capacity", 100000),
        device=CONFIG.get("device", "cuda")
    )
    agent = DQNAgent(
        input_shape=input_shape,
        n_actions=n_actions,
        device=CONFIG.get("device", "cpu"),
        config=CONFIG,
        memory=memory
    )

    # Reduce the LR on the fly
    agent.optimizer.param_groups[0]['lr'] = 3e-5

    # attach AMP learn if enabled
    if CONFIG.get("use_amp") and CONFIG.get("device") == "cuda":
        from amp_learn_function import learn_amp
        agent.learn_amp = learn_amp.__get__(agent, type(agent))
        logger.info("AMP enabled for training.")

    # debug inspect learn methods
    logger.info("=== DEBUG: learn methods ===")
    logger.info(f"learn(): {hasattr(agent, 'learn')}")
    logger.info(f"learn_amp(): {hasattr(agent, 'learn_amp')}")
    logger.info("=== END DEBUG ===")

    # register checkpoint on exit
    checkpoint_path = os.path.join(CONFIG.get("checkpoint_dir", "checkpoints"), "checkpoint_latest.pt")
    atexit.register(save_checkpoint_on_exit, agent, checkpoint_path, logger)

    # graceful signal handlers
    def _handler(sig, frame):
        save_checkpoint_on_exit(agent, checkpoint_path, logger)
        sys.exit(0)
    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)

    # start training
    logger.info("\nStarting training…")
    from gpu_train import train_gpu_optimized
    train_gpu_optimized(env, agent, config=CONFIG, logger=logger)
    env.close()


if __name__ == "__main__":
    main()