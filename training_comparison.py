#!/usr/bin/env python
"""
Script to compare training performance between standard low-level actions
and high-level action variants of the Tetris DQN.
"""
import subprocess
import os
import time
import argparse
import matplotlib.pyplot as plt
import numpy as np
import re

def parse_args():
    parser = argparse.ArgumentParser(description="Compare low-level vs high-level action training")
    parser.add_argument("--episodes", type=int, default=1000,
                        help="Number of episodes for each training run")
    parser.add_argument("--parallel-envs", type=int, default=8,
                        help="Number of parallel environments to use")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size for training")
    parser.add_argument("--no-analysis", action="store_true",
                        help="Skip analysis step")
    return parser.parse_args()

def run_training_test(args):
    """Run a comparison between low-level and high-level action training."""
    # Parameters from args
    episodes = args.episodes
    parallel_envs = args.parallel_envs
    batch_size = args.batch_size
    
    # Directory for results
    os.makedirs("comparison_results", exist_ok=True)
    
    # Run low-level training
    print("\n=== RUNNING LOW-LEVEL ACTION TRAINING ===")
    print(f"Episodes: {episodes}, Parallel Envs: {parallel_envs}, Batch Size: {batch_size}")
    
    low_level_start = time.time()
    low_level_cmd = [
        "python", "run_gpu_training_updated.py",
        "--episodes", str(episodes),
        "--parallel-envs", str(parallel_envs),
        "--batch-size", str(batch_size),
        "--log-file", "comparison_results/low_level_training.log"
    ]
    subprocess.run(low_level_cmd)
    low_level_time = time.time() - low_level_start
    
    # Run high-level training
    print("\n=== RUNNING HIGH-LEVEL ACTION TRAINING ===")
    print(f"Episodes: {episodes}, Parallel Envs: {parallel_envs}, Batch Size: {batch_size}")
    
    high_level_start = time.time()
    high_level_cmd = [
        "python", "run_gpu_training_updated.py",
        "--episodes", str(episodes),
        "--parallel-envs", str(parallel_envs),
        "--batch-size", str(batch_size),
        "--high-level-actions",
        "--log-file", "comparison_results/high_level_training.log"
    ]
    subprocess.run(high_level_cmd)
    high_level_time = time.time() - high_level_start
    
    # Write timing information
    with open("comparison_results/timing_info.txt", "w") as f:
        f.write(f"Low-level training time: {low_level_time:.2f} seconds ({low_level_time/3600:.2f} hours)\n")
        f.write(f"High-level training time: {high_level_time:.2f} seconds ({high_level_time/3600:.2f} hours)\n")
        f.write(f"Speed difference: {low_level_time/high_level_time:.2f}x\n")
    
    print("\n=== TRAINING TIMING COMPARISON ===")
    print(f"Low-level training time: {low_level_time:.2f} seconds ({low_level_time/3600:.2f} hours)")
    print(f"High-level training time: {high_level_time:.2f} seconds ({high_level_time/3600:.2f} hours)")
    print(f"Speed difference: {low_level_time/high_level_time:.2f}x")
    
    # Analyze results if requested
    if not args.no_analysis:
        print("\n=== ANALYZING RESULTS ===")
        analyze_cmd = [
            "python", "analyze_dqn_log.py",
            "comparison_results/low_level_training.log",
            "--output", "comparison_results/low_level_analysis.txt",
            "--plots"
        ]
        subprocess.run(analyze_cmd)
        
        analyze_cmd = [
            "python", "analyze_dqn_log.py",
            "comparison_results/high_level_training.log",
            "--output", "comparison_results/high_level_analysis.txt",
            "--plots"
        ]
        subprocess.run(analyze_cmd)
        
        # Generate comparison plots
        generate_comparison_plots()
    
    print("\nComparison complete! Check the comparison_results directory for analysis.")

def extract_episode_data(log_file):
    """Extract episode rewards and lines cleared from log file."""
    try:
        with open(log_file, 'r') as f:
            content = f.read()
        
        # Extract episode data
        episode_pattern = r"Episode (\d+)/\d+, Steps: \d+, Reward: ([-\d\.]+), Avg\(\d+\): ([-\d\.]+)"
        episode_matches = re.findall(episode_pattern, content)
        
        # Extract line clearing data
        line_clear_pattern = r"LINE CLEAR REWARD: Base=[\d\.]+, for (\d+) lines"
        line_matches = re.findall(line_clear_pattern, content)
        
        # Process episode data
        episodes = []
        rewards = []
        avg_rewards = []
        
        for ep, reward, avg_reward in episode_matches:
            episodes.append(int(ep))
            rewards.append(float(reward))
            avg_rewards.append(float(avg_reward))
        
        # Count lines cleared
        total_lines = sum(int(lines) for lines in line_matches)
        
        return {
            'episodes': episodes,
            'rewards': rewards,
            'avg_rewards': avg_rewards,
            'total_lines': total_lines
        }
    except Exception as e:
        print(f"Error extracting data from {log_file}: {e}")
        return {
            'episodes': [],
            'rewards': [],
            'avg_rewards': [],
            'total_lines': 0
        }

def generate_comparison_plots():
    """Generate plots comparing low-level and high-level training results."""
    # Load data
    low_level_data = extract_episode_data("comparison_results/low_level_training.log")
    high_level_data = extract_episode_data("comparison_results/high_level_training.log")
    
    # Create comparison directory
    os.makedirs("comparison_results/comparison_plots", exist_ok=True)
    
    # Plot episode rewards
    plt.figure(figsize=(12, 6))
    if low_level_data['episodes']:
        plt.plot(low_level_data['episodes'], low_level_data['rewards'], label='Low-Level Actions', alpha=0.7)
    if high_level_data['episodes']:
        plt.plot(high_level_data['episodes'], high_level_data['rewards'], label='High-Level Actions', alpha=0.7)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Episode Rewards: Low-Level vs High-Level Actions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("comparison_results/comparison_plots/reward_comparison.png")
    
    # Plot average rewards
    plt.figure(figsize=(12, 6))
    if low_level_data['episodes']:
        plt.plot(low_level_data['episodes'], low_level_data['avg_rewards'], label='Low-Level Actions', alpha=0.7)
    if high_level_data['episodes']:
        plt.plot(high_level_data['episodes'], high_level_data['avg_rewards'], label='High-Level Actions', alpha=0.7)
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title('Moving Average Rewards: Low-Level vs High-Level Actions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("comparison_results/comparison_plots/avg_reward_comparison.png")
    
    # Create summary text file
    with open("comparison_results/comparison_summary.txt", "w") as f:
        f.write("=== TRAINING COMPARISON SUMMARY ===\n\n")
        
        # Episode information
        low_episodes = len(low_level_data['episodes'])
        high_episodes = len(high_level_data['episodes'])
        f.write(f"Low-level episodes completed: {low_episodes}\n")
        f.write(f"High-level episodes completed: {high_episodes}\n\n")
        
        # Reward information
        if low_level_data['rewards']:
            low_max_reward = max(low_level_data['rewards'])
            low_avg_reward = sum(low_level_data['rewards']) / len(low_level_data['rewards'])
            f.write(f"Low-level max reward: {low_max_reward:.2f}\n")
            f.write(f"Low-level average reward: {low_avg_reward:.2f}\n")
        
        if high_level_data['rewards']:
            high_max_reward = max(high_level_data['rewards'])
            high_avg_reward = sum(high_level_data['rewards']) / len(high_level_data['rewards'])
            f.write(f"High-level max reward: {high_max_reward:.2f}\n")
            f.write(f"High-level average reward: {high_avg_reward:.2f}\n\n")
        
        # Lines cleared information
        f.write(f"Low-level total lines cleared: {low_level_data['total_lines']}\n")
        f.write(f"High-level total lines cleared: {high_level_data['total_lines']}\n")
        
        if low_episodes > 0 and high_episodes > 0:
            f.write(f"Low-level lines per episode: {low_level_data['total_lines']/low_episodes:.2f}\n")
            f.write(f"High-level lines per episode: {high_level_data['total_lines']/high_episodes:.2f}\n")
            
            if low_level_data['total_lines'] > 0 and high_level_data['total_lines'] > 0:
                f.write(f"\nImprovement ratio: {high_level_data['total_lines']/low_level_data['total_lines']:.2f}x more lines cleared with high-level actions\n")

if __name__ == "__main__":
    args = parse_args()
    run_training_test(args)