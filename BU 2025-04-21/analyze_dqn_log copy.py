#!/usr/bin/env python
import re
import sys
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze DQN Tetris training logs')
    parser.add_argument('log_file', type=str, help='Path to the log file')
    parser.add_argument('--output', type=str, default='training_analysis.txt',
                      help='Output file for summary (default: training_analysis.txt)')
    parser.add_argument('--plots', action='store_true', help='Generate plots')
    return parser.parse_args()

def extract_learn_step_data(log_content):
    """Extract data from learn steps in the log file."""
    # Regular expressions to extract values
    learn_step_pattern = r"--- Learn Step (\d+) ---"
    reward_pattern = r"Sampled Rewards: min=([-\d\.]+), max=([-\d\.]+), mean=([-\d\.]+)"
    q_value_pattern = r"Q\(s,a\): min=([-\d\.]+), max=([-\d\.]+), mean=([-\d\.]+)"
    expected_q_pattern = r"Expected Q: min=([-\d\.]+), max=([-\d\.]+), mean=([-\d\.]+)"
    loss_pattern = r"Loss = ([\d\.]+), Avg Q = ([-\d\.]+)"
    grad_norm_pattern = r"Grad Norm = ([\d\.]+)"
    
    # Initialize data structures
    data = {
        'step': [],
        'loss': [],
        'avg_q': [],
        'q_min': [],
        'q_max': [],
        'q_mean': [],
        'expected_q_min': [],
        'expected_q_max': [],
        'expected_q_mean': [],
        'reward_min': [],
        'reward_max': [],
        'reward_mean': [],
        'grad_norm': []
    }
    
    # Extract data using regular expressions
    lines = log_content.split('\n')
    current_step = None
    
    for line in lines:
        # Extract learn step number
        step_match = re.search(learn_step_pattern, line)
        if step_match:
            current_step = int(step_match.group(1))
            data['step'].append(current_step)
            continue
            
        # Only process metrics if we have a current step
        if current_step is None:
            continue
            
        # Extract loss and avg_q
        loss_match = re.search(loss_pattern, line)
        if loss_match:
            data['loss'].append(float(loss_match.group(1)))
            data['avg_q'].append(float(loss_match.group(2)))
            continue
            
        # Extract Q-values
        q_match = re.search(q_value_pattern, line)
        if q_match:
            data['q_min'].append(float(q_match.group(1)))
            data['q_max'].append(float(q_match.group(2)))
            data['q_mean'].append(float(q_match.group(3)))
            continue
            
        # Extract expected Q-values
        expected_q_match = re.search(expected_q_pattern, line)
        if expected_q_match:
            data['expected_q_min'].append(float(expected_q_match.group(1)))
            data['expected_q_max'].append(float(expected_q_match.group(2)))
            data['expected_q_mean'].append(float(expected_q_match.group(3)))
            continue
            
        # Extract rewards
        reward_match = re.search(reward_pattern, line)
        if reward_match:
            data['reward_min'].append(float(reward_match.group(1)))
            data['reward_max'].append(float(reward_match.group(2)))
            data['reward_mean'].append(float(reward_match.group(3)))
            continue
            
        # Extract gradient norm
        grad_match = re.search(grad_norm_pattern, line)
        if grad_match:
            data['grad_norm'].append(float(grad_match.group(1)))
            continue
    
    # Make sure all lists are the same length by using the shortest length
    min_length = min(len(data[key]) for key in data.keys() if data[key])
    for key in data.keys():
        if data[key]:
            data[key] = data[key][:min_length]
    
    return data

def extract_episode_data(log_content):
    """Extract episode-related data from the log."""
    episode_pattern = r"Episode (\d+)/\d+, Steps: (\d+), Reward: ([-\d\.]+), Avg\((\d+)\): ([-\d\.]+)"
    
    data = {
        'episode': [],
        'steps': [],
        'reward': [],
        'avg_window': [],
        'avg_reward': []
    }
    
    # Extract episode data
    for line in log_content.split('\n'):
        match = re.search(episode_pattern, line)
        if match:
            data['episode'].append(int(match.group(1)))
            data['steps'].append(int(match.group(2)))
            data['reward'].append(float(match.group(3)))
            data['avg_window'].append(int(match.group(4)))
            data['avg_reward'].append(float(match.group(5)))
    
    return data

def extract_config_data(log_content):
    """Extract configuration data from the log."""
    config = {}
    
    # Extract basic configuration
    config['episodes'] = re.search(r"Episodes: (\d+)", log_content)
    config['episodes'] = int(config['episodes'].group(1)) if config['episodes'] else None
    
    config['batch_size'] = re.search(r"Batch size: (\d+)", log_content)
    config['batch_size'] = int(config['batch_size'].group(1)) if config['batch_size'] else None
    
    config['curriculum'] = re.search(r"Curriculum: (\w+)", log_content)
    config['curriculum'] = config['curriculum'].group(1) if config['curriculum'] else None
    
    config['parallel_envs'] = re.search(r"Parallel envs: (\d+)", log_content)
    config['parallel_envs'] = int(config['parallel_envs'].group(1)) if config['parallel_envs'] else None
    
    config['learning_rate'] = re.search(r"Learning rate \(optimizer\): ([\d\.e-]+)", log_content)
    config['learning_rate'] = float(config['learning_rate'].group(1)) if config['learning_rate'] else None
    
    config['model_type'] = re.search(r"model type: (\w+)", log_content)
    config['model_type'] = config['model_type'].group(1) if config['model_type'] else None
    
    return config

def analyze_training(learn_data, episode_data, config):
    """Generate analysis of the training data."""
    analysis = []
    
    # Configuration summary
    analysis.append("=== DQN TRAINING ANALYSIS ===")
    analysis.append(f"Model Type: {config.get('model_type', 'Unknown')}")
    analysis.append(f"Total Episodes: {config.get('episodes', 'Unknown')}")
    analysis.append(f"Batch Size: {config.get('batch_size', 'Unknown')}")
    analysis.append(f"Parallel Environments: {config.get('parallel_envs', 'Unknown')}")
    analysis.append(f"Learning Rate: {config.get('learning_rate', 'Unknown')}")
    analysis.append(f"Curriculum Learning: {config.get('curriculum', 'Unknown')}")
    
    # Learning process summary
    if learn_data['step']:
        max_steps = max(learn_data['step'])
        analysis.append(f"\n=== LEARNING PROGRESS (Total Steps: {max_steps}) ===")
        
        # Loss analysis
        if learn_data['loss']:
            early_loss_avg = np.mean(learn_data['loss'][:min(10, len(learn_data['loss']))])
            late_loss_avg = np.mean(learn_data['loss'][-min(10, len(learn_data['loss'])):])
            loss_trend = "DECREASING" if late_loss_avg < early_loss_avg else "INCREASING"
            analysis.append(f"Loss: Early avg={early_loss_avg:.3f}, Late avg={late_loss_avg:.3f}, Trend: {loss_trend}")
        
        # Q-value analysis
        if learn_data['avg_q']:
            early_q_avg = np.mean(learn_data['avg_q'][:min(10, len(learn_data['avg_q']))])
            late_q_avg = np.mean(learn_data['avg_q'][-min(10, len(learn_data['avg_q'])):])
            q_trend = "INCREASING" if late_q_avg > early_q_avg else "DECREASING"
            analysis.append(f"Avg Q-value: Early avg={early_q_avg:.3f}, Late avg={late_q_avg:.3f}, Trend: {q_trend}")
        
        # Reward analysis
        if learn_data['reward_mean']:
            early_reward_avg = np.mean(learn_data['reward_mean'][:min(10, len(learn_data['reward_mean']))])
            late_reward_avg = np.mean(learn_data['reward_mean'][-min(10, len(learn_data['reward_mean'])):])
            reward_trend = "INCREASING" if late_reward_avg > early_reward_avg else "DECREASING"
            analysis.append(f"Batch Rewards: Early avg={early_reward_avg:.3f}, Late avg={late_reward_avg:.3f}, Trend: {reward_trend}")
        
        # Gradient norm analysis
        if learn_data['grad_norm']:
            grad_norm_avg = np.mean(learn_data['grad_norm'])
            grad_norm_max = np.max(learn_data['grad_norm'])
            grad_norm_recent = np.mean(learn_data['grad_norm'][-min(20, len(learn_data['grad_norm'])):])
            analysis.append(f"Gradient Norm: Avg={grad_norm_avg:.2f}, Max={grad_norm_max:.2f}, Recent avg={grad_norm_recent:.2f}")
            if grad_norm_avg > 50:
                analysis.append("WARNING: Average gradient norm is very high (>50). Consider reducing learning rate.")
    
    # Episode performance summary
    if episode_data['episode']:
        analysis.append(f"\n=== EPISODE PERFORMANCE (Episodes: {len(episode_data['episode'])}) ===")
        
        if episode_data['reward']:
            min_reward = min(episode_data['reward'])
            max_reward = max(episode_data['reward'])
            avg_reward = np.mean(episode_data['reward'])
            analysis.append(f"Episode Rewards: Min={min_reward:.2f}, Max={max_reward:.2f}, Avg={avg_reward:.2f}")
        
        if episode_data['steps']:
            min_steps = min(episode_data['steps'])
            max_steps = max(episode_data['steps'])
            avg_steps = np.mean(episode_data['steps'])
            analysis.append(f"Episode Steps: Min={min_steps}, Max={max_steps}, Avg={avg_steps:.2f}")
        
        # Detect if reward is improving
        if len(episode_data['reward']) > 5:
            early_episodes = episode_data['reward'][:len(episode_data['reward'])//2]
            late_episodes = episode_data['reward'][len(episode_data['reward'])//2:]
            early_avg = np.mean(early_episodes)
            late_avg = np.mean(late_episodes)
            if late_avg > early_avg:
                analysis.append(f"POSITIVE: Average reward is improving (Early: {early_avg:.2f}, Late: {late_avg:.2f})")
            else:
                analysis.append(f"CONCERN: Average reward is not improving (Early: {early_avg:.2f}, Late: {late_avg:.2f})")
    
    # Health check and recommendations
    analysis.append("\n=== TRAINING HEALTH CHECK ===")
    
    # Q-value health check
    if learn_data['q_mean']:
        q_mean_recent = np.mean(learn_data['q_mean'][-min(20, len(learn_data['q_mean'])):])
        if q_mean_recent < -10:
            analysis.append("WARNING: Recent Q-values are very negative. Agent may be developing a pessimistic policy.")
        elif q_mean_recent > 50:
            analysis.append("WARNING: Recent Q-values are very high. Potential value overestimation.")
    
    # Loss stability check
    if learn_data['loss'] and len(learn_data['loss']) > 10:
        loss_std = np.std(learn_data['loss'][-20:])
        loss_mean = np.mean(learn_data['loss'][-20:])
        loss_cv = loss_std / loss_mean if loss_mean > 0 else 0
        if loss_cv > 0.5:
            analysis.append(f"WARNING: Loss is unstable (coefficient of variation: {loss_cv:.2f}). Consider adjusting learning rate.")
    
    # Gradient norm stability check
    if learn_data['grad_norm'] and len(learn_data['grad_norm']) > 10:
        if np.mean(learn_data['grad_norm'][-20:]) > 40:
            analysis.append("WARNING: Recent gradient norms are high. Consider reducing learning rate.")
    
    # Recommendations
    analysis.append("\n=== RECOMMENDATIONS ===")
    
    # Check if reward is improving
    if episode_data['reward'] and len(episode_data['reward']) > 5:
        if np.mean(episode_data['reward'][-5:]) < np.mean(episode_data['reward'][:5]):
            analysis.append("1. Agent is not improving. Consider revising reward function or exploring more.")
    
    # Check for potential exploration issues
    if episode_data['steps'] and max(episode_data['steps']) < 100:
        analysis.append("2. Episodes are very short. Agent may not be exploring effectively.")
    
    # Check for gradient issues
    if learn_data['grad_norm'] and np.mean(learn_data['grad_norm']) > 30:
        analysis.append("3. Gradient norms are high. Consider reducing learning rate or implementing gradient clipping.")
    
    # Check for value function issues
    if learn_data['q_mean'] and np.mean(learn_data['q_mean'][-min(20, len(learn_data['q_mean'])):]) < -20:
        analysis.append("4. Q-values are very negative. Consider scaling rewards or adjusting discount factor.")
    
    return analysis

def generate_plots(learn_data, episode_data, output_prefix):
    """Generate plots of training metrics."""
    if not (learn_data['step'] and learn_data['loss']):
        print("Not enough data to generate plots.")
        return
    
    # Directory for plots
    plots_dir = f"{output_prefix}_plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot 1: Loss over time
    plt.figure(figsize=(10, 6))
    plt.plot(learn_data['step'], learn_data['loss'])
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Loss Over Training')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/loss.png")
    plt.close()
    
    # Plot 2: Q-values over time
    plt.figure(figsize=(10, 6))
    plt.plot(learn_data['step'], learn_data['q_mean'], label='Mean Q-value')
    if learn_data['avg_q']:
        plt.plot(learn_data['step'], learn_data['avg_q'], label='Avg Q (selected actions)')
    plt.xlabel('Training Steps')
    plt.ylabel('Q-value')
    plt.title('Q-values Over Training')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/q_values.png")
    plt.close()
    
    # Plot 3: Gradient norms over time
    if learn_data['grad_norm']:
        plt.figure(figsize=(10, 6))
        plt.plot(learn_data['step'], learn_data['grad_norm'])
        plt.xlabel('Training Steps')
        plt.ylabel('Gradient Norm')
        plt.title('Gradient Norms Over Training')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{plots_dir}/grad_norms.png")
        plt.close()
    
    # Plot 4: Episode rewards
    if episode_data['episode'] and episode_data['reward']:
        plt.figure(figsize=(10, 6))
        plt.plot(episode_data['episode'], episode_data['reward'], label='Episode Reward')
        if episode_data['avg_reward']:
            plt.plot(episode_data['episode'], episode_data['avg_reward'], 
                     label=f'Moving Avg (window={episode_data["avg_window"][0]})')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Episode Rewards')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{plots_dir}/episode_rewards.png")
        plt.close()
    
    # Plot 5: Episode lengths
    if episode_data['episode'] and episode_data['steps']:
        plt.figure(figsize=(10, 6))
        plt.plot(episode_data['episode'], episode_data['steps'])
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        plt.title('Episode Lengths')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{plots_dir}/episode_lengths.png")
        plt.close()
    
    print(f"Plots saved to {plots_dir}/")

def main():
    args = parse_args()
    
    # Read log file
    try:
        with open(args.log_file, 'r') as f:
            log_content = f.read()
    except Exception as e:
        print(f"Error reading log file: {e}")
        return
    
    # Extract data
    learn_data = extract_learn_step_data(log_content)
    episode_data = extract_episode_data(log_content)
    config_data = extract_config_data(log_content)
    
    # Analyze data
    analysis = analyze_training(learn_data, episode_data, config_data)
    
    # Write analysis to file
    with open(args.output, 'w') as f:
        f.write('\n'.join(analysis))
    
    print(f"Analysis written to {args.output}")
    
    # Print summary to console
    for line in analysis:
        print(line)
    
    # Generate plots if requested
    if args.plots:
        output_prefix = os.path.splitext(args.output)[0]
        generate_plots(learn_data, episode_data, output_prefix)

if __name__ == '__main__':
    main()