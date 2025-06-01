#/usr/bin/env python3
"""
Generated training script for enhanced configuration.
Episodes: 1000, Device: cpu
"""
import sys
import os
import time
import torch
import numpy as np

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simple_tetris_env import SimpleTetrisEnv
from agent import DQNAgent
from preprocessing import preprocess_state
from minimal_config import get_minimal_config, get_enhanced_config, get_gpu_config, validate_config

def main():
    print("=" * 60)
    print(f"TETRIS DQN TRAINING - enhanced.upper^(^) CONFIGURATION")
    print("=" * 60)
    print(f"Episodes: 1000")
    print(f"Device: cpu")
    print(f"Time: {time.strftime^('%Y-%m-%d %H:%M:%S'^)}")
    print()
ECHO is off.
    # Load configuration
    if "enhanced" == "minimal":
        config = get_minimal_config()
    elif "enhanced" == "enhanced":
        config = get_enhanced_config()
    elif "enhanced" == "gpu":
        config = get_gpu_config()
    else:
        raise ValueError(f"Unknown config type: enhanced")
ECHO is off.
    # Override settings
    config["num_episodes"] = 1000
    config["device"] = "cpu"
ECHO is off.
    # Validate configuration
    config = validate_config(config)
ECHO is off.
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
ECHO is off.
    # Create environment
    env = SimpleTetrisEnv()
ECHO is off.
    # Determine input shape based on preprocessing
    if config.get("use_enhanced_preprocessing", False):
        input_shape = (4, 14, 7)
    else:
        input_shape = (1, 14, 7)
ECHO is off.
    # Create agent
    agent = DQNAgent(
        input_shape=input_shape,
        n_actions=env.action_space.n,
        device=config["device"],
        config=config
    )
ECHO is off.
    print(f"Agent created with {sum^(p.numel^(^) for p in agent.policy_net.parameters^(^)^)} parameters")
    print()
ECHO is off.
    # Training loop
    episode_rewards = []
    episode_lengths = []
    lines_cleared_total = []
ECHO is off.
    start_time = time.time()
ECHO is off.
    for episode in range(config["num_episodes"]):
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        episode_lines_cleared = 0
ECHO is off.
        while True:
            # Preprocess state
            processed_state = preprocess_state(
                state, 
                include_piece_info=config.get("use_enhanced_preprocessing", False),
                device=config["device"]
            )
ECHO is off.
            # Select action
            action = agent.select_action(processed_state, training=True)
ECHO is off.
            # Environment step
            next_state, reward, done, info = env.step(action)
ECHO is off.
            # Store transition
            if next_state is not None:
                next_processed_state = preprocess_state(
                    next_state,
                    include_piece_info=config.get("use_enhanced_preprocessing", False),
                    device=config["device"]
                )
            else:
                next_processed_state = None
ECHO is off.
            agent.memory.push(processed_state, action, next_processed_state, reward, done)
ECHO is off.
            # Learn
            if len(agent.memory) >= agent.batch_size:
                loss = agent.learn()
ECHO is off.
            # Update metrics
            episode_reward += reward
            episode_length += 1
ECHO is off.
            if 'lines_cleared' in info:
                episode_lines_cleared += info['lines_cleared']
ECHO is off.
            if done:
                break
ECHO is off.
            state = next_state
ECHO is off.
        # Record episode metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        lines_cleared_total.append(episode_lines_cleared)
ECHO is off.
        # Update exploration
        agent.epsilon = max(agent.epsilon_end, agent.epsilon * agent.epsilon_decay)
ECHO is off.
        # Logging
        if episode % max(1, config["num_episodes"] // 10) == 0 or episode == config["num_episodes"] - 1:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_length = np.mean(episode_lengths[-100:])
            avg_lines = np.mean(lines_cleared_total[-100:])
            elapsed = time.time() - start_time
ECHO is off.
            print(f"Episode {episode:4d} | "
                  f"Reward: {episode_reward:6.1f} | "
                  f"Avg^(100^): {avg_reward:6.1f} | "
                  f"Length: {episode_length:3d} | "
                  f"Lines: {episode_lines_cleared:2d} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Time: {elapsed:.0f}s")
ECHO is off.
    # Final statistics
    print()
    print("=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)
    print(f"Total episodes: {len^(episode_rewards^)}")
    print(f"Average reward: {np.mean^(episode_rewards^):.2f}")
    print(f"Best episode reward: {np.max^(episode_rewards^):.2f}")
    print(f"Total lines cleared: {np.sum^(lines_cleared_total^)}")
    print(f"Average lines per episode: {np.mean^(lines_cleared_total^):.2f}")
    print(f"Training time: {^(time.time^(^) - start_time^) / 60:.1f} minutes")
ECHO is off.
    # Save model
    model_path = f"tetris_dqn_enhanced_1000ep.pt"
    agent.save(model_path)
    print(f"Model saved to: {model_path}")
ECHO is off.
    return {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'lines_cleared': lines_cleared_total,
        'config': config
    }

if __name__ == "__main__":
    try:
        result = main()
        print("\n✅ Training completed successfully")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
