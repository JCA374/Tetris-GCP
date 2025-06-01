#!/usr/bin/env python
"""
Script to evaluate and compare trained Tetris DQN models with low-level and high-level actions.
"""
import os
import argparse
import numpy as np
import torch
from simple_tetris_env import SimpleTetrisEnv
from high_level_env import HighLevelTetrisEnv, visualize_placements
from agent import DQNAgent
from train import preprocess_state
from config import CONFIG
import time

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained Tetris DQN models")
    parser.add_argument("--low-level-model", type=str, default="models/low_level_final_model.pt",
                        help="Path to low-level action model file")
    parser.add_argument("--high-level-model", type=str, default="models/high_level_final_model.pt",
                        help="Path to high-level action model file")
    parser.add_argument("--episodes", type=int, default=20,
                        help="Number of episodes to evaluate for each model")
    parser.add_argument("--render", action="store_true",
                        help="Render the game during evaluation")
    parser.add_argument("--visualize-high-level", action="store_true",
                        help="Visualize high-level action possibilities")
    parser.add_argument("--max-steps", type=int, default=10000,
                        help="Maximum steps per episode")
    parser.add_argument("--width", type=int, default=7,
                        help="Grid width")
    parser.add_argument("--height", type=int, default=14,
                        help="Grid height")
    return parser.parse_args()

def evaluate_model(model_path, env, episodes=10, max_steps=5000, render=False, 
                   use_enhanced_preprocessing=True, is_high_level=False, visualize=False):
    """Evaluate a trained DQN model."""
    print(f"\nEvaluating model: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found.")
        return None
    
    # Determine environment shape and action space
    state = env.reset()
    processed_state = preprocess_state(
        state, 
        binary=CONFIG.get("use_binary_states", False),
        include_piece_info=use_enhanced_preprocessing
    )
    input_shape = processed_state.shape
    n_actions = env.action_space.n
    
    print(f"Input shape: {input_shape}, Action space: {n_actions}")
    
    # Create agent with appropriate action space
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = DQNAgent(
        input_shape=input_shape,
        n_actions=n_actions,
        device=device,
        config=CONFIG
    )
    
    # Load model
    loaded_config = agent.load(model_path)
    print(f"Model loaded from {model_path}")
    
    # Statistics
    rewards = []
    steps_list = []
    lines_cleared_list = []
    game_over_count = 0
    board_heights = []
    holes_counts = []
    
    render_delay = 0.05 if render else 0
    
    # Evaluate for specified number of episodes
    for episode in range(episodes):
        state = env.reset()
        processed_state = preprocess_state(
            state, 
            binary=CONFIG.get("use_binary_states", False),
            include_piece_info=use_enhanced_preprocessing
        )
        
        if render:
            env.render()
            time.sleep(render_delay)
            
        if visualize and is_high_level:
            print(f"\nEpisode {episode+1} - High-Level Action Options:")
            viz_grids = visualize_placements(env, top_k=3)
            if viz_grids:
                for i, grid in enumerate(viz_grids):
                    print(f"Option {i+1}:")
                    for row in grid:
                        print("".join(str(int(cell)) if cell > 0 else "." for cell in row))
        
        episode_reward = 0
        episode_steps = 0
        episode_lines = 0
        done = False
        
        # Keep track of last 5 actions for loop detection
        last_actions = []
        
        for step in range(max_steps):
            # Select action deterministically (no exploration)
            action = agent.get_action(processed_state, deterministic=True)
            
            # Track actions for loop detection
            last_actions.append(action)
            if len(last_actions) > 5:
                last_actions.pop(0)
                
            # Detect if agent is stuck in an action loop
            if len(last_actions) == 5 and len(set(last_actions)) == 1 and step > 100:
                print(f"Episode {episode+1} detected action loop - forcing termination")
                break
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            if render:
                env.render()
                time.sleep(render_delay)
                
            # Track high-level placement if available
            if is_high_level and 'placement' in info:
                placement = info['placement']
                if placement:
                    rotation, x, y = placement
                    if visualize:
                        print(f"Step {step}: Placed piece at rotation={rotation}, x={x}, y={y}, reward={reward:.2f}")
            
            # Update metrics
            episode_reward += reward
            episode_steps += 1
            episode_lines += info.get('lines_cleared', 0)
            
            # Update state
            if not done:
                processed_state = preprocess_state(
                    next_state, 
                    binary=CONFIG.get("use_binary_states", False),
                    include_piece_info=use_enhanced_preprocessing
                )
            else:
                game_over_count += 1
                break
        
        # Collect metrics
        rewards.append(episode_reward)
        steps_list.append(episode_steps)
        lines_cleared_list.append(episode_lines)
        
        # Get final board state metrics
        if hasattr(env, 'env'):
            # Handle wrapped environment
            board_heights.append(env.env._get_board_height())
            holes_counts.append(env.env._count_holes())
        else:
            # Direct environment
            board_heights.append(env._get_board_height())
            holes_counts.append(env._count_holes())
        
        print(f"Episode {episode+1}: Reward={episode_reward:.2f}, Steps={episode_steps}, Lines={episode_lines}")
    
    # Calculate overall statistics
    avg_reward = sum(rewards) / len(rewards) if rewards else 0
    avg_steps = sum(steps_list) / len(steps_list) if steps_list else 0
    avg_lines = sum(lines_cleared_list) / len(lines_cleared_list) if lines_cleared_list else 0
    total_lines = sum(lines_cleared_list)
    avg_height = sum(board_heights) / len(board_heights) if board_heights else 0
    avg_holes = sum(holes_counts) / len(holes_counts) if holes_counts else 0
    
    # Display results
    print("\nEvaluation Results:")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Steps: {avg_steps:.2f}")
    print(f"Average Lines Cleared: {avg_lines:.2f}")
    print(f"Total Lines Cleared: {total_lines}")
    print(f"Game Over Rate: {game_over_count/episodes:.2f}")
    print(f"Average Final Board Height: {avg_height:.2f}")
    print(f"Average Final Holes: {avg_holes:.2f}")
    
    # Return evaluation results
    return {
        "avg_reward": avg_reward,
        "avg_steps": avg_steps,
        "avg_lines": avg_lines,
        "total_lines": total_lines,
        "game_over_rate": game_over_count/episodes,
        "avg_height": avg_height,
        "avg_holes": avg_holes
    }

def main():
    args = parse_args()
    
    # Create directory for comparison results
    os.makedirs("evaluation_results", exist_ok=True)
    
    # Create environments
    low_level_env = SimpleTetrisEnv(grid_width=args.width, grid_height=args.height)
    high_level_env = HighLevelTetrisEnv(grid_width=args.width, grid_height=args.height)
    
    # Evaluate low-level model if file exists
    low_level_results = None
    if os.path.exists(args.low_level_model):
        low_level_results = evaluate_model(
            args.low_level_model,
            low_level_env,
            episodes=args.episodes,
            max_steps=args.max_steps,
            render=args.render,
            is_high_level=False
        )
    else:
        print(f"Low-level model file {args.low_level_model} not found.")
    
    # Evaluate high-level model if file exists
    high_level_results = None
    if os.path.exists(args.high_level_model):
        high_level_results = evaluate_model(
            args.high_level_model,
            high_level_env,
            episodes=args.episodes,
            max_steps=args.max_steps,
            render=args.render,
            is_high_level=True,
            visualize=args.visualize_high_level
        )
    else:
        print(f"High-level model file {args.high_level_model} not found.")
    
    # Compare results if both models were evaluated
    if low_level_results and high_level_results:
        print("\n=== MODEL COMPARISON ===")
        
        # Calculate improvement ratios
        lines_ratio = high_level_results["total_lines"] / max(1, low_level_results["total_lines"])
        reward_ratio = high_level_results["avg_reward"] / max(1, low_level_results["avg_reward"]) if low_level_results["avg_reward"] != 0 else float('inf')
        steps_ratio = high_level_results["avg_steps"] / max(1, low_level_results["avg_steps"]) if low_level_results["avg_steps"] != 0 else float('inf')
        
        print(f"Lines cleared improvement: {lines_ratio:.2f}x")
        print(f"Reward improvement: {reward_ratio:.2f}x")
        print(f"Survival time improvement: {steps_ratio:.2f}x")
        
        # Write comparison to file
        with open("evaluation_results/model_comparison.txt", "w") as f:
            f.write("=== TETRIS DQN MODEL COMPARISON ===\n\n")
            
            f.write("=== LOW-LEVEL ACTION MODEL ===\n")
            f.write(f"Average Reward: {low_level_results['avg_reward']:.2f}\n")
            f.write(f"Average Steps: {low_level_results['avg_steps']:.2f}\n")
            f.write(f"Average Lines Cleared: {low_level_results['avg_lines']:.2f}\n")
            f.write(f"Total Lines Cleared: {low_level_results['total_lines']}\n")
            f.write(f"Game Over Rate: {low_level_results['game_over_rate']:.2f}\n")
            f.write(f"Average Final Board Height: {low_level_results['avg_height']:.2f}\n")
            f.write(f"Average Final Holes: {low_level_results['avg_holes']:.2f}\n\n")
            
            f.write("=== HIGH-LEVEL ACTION MODEL ===\n")
            f.write(f"Average Reward: {high_level_results['avg_reward']:.2f}\n")
            f.write(f"Average Steps: {high_level_results['avg_steps']:.2f}\n")
            f.write(f"Average Lines Cleared: {high_level_results['avg_lines']:.2f}\n")
            f.write(f"Total Lines Cleared: {high_level_results['total_lines']}\n")
            f.write(f"Game Over Rate: {high_level_results['game_over_rate']:.2f}\n")
            f.write(f"Average Final Board Height: {high_level_results['avg_height']:.2f}\n")
            f.write(f"Average Final Holes: {high_level_results['avg_holes']:.2f}\n\n")
            
            f.write("=== IMPROVEMENT RATIOS ===\n")
            f.write(f"Lines cleared improvement: {lines_ratio:.2f}x\n")
            f.write(f"Reward improvement: {reward_ratio:.2f}x\n")
            f.write(f"Survival time improvement: {steps_ratio:.2f}x\n")
        
        print(f"Comparison saved to evaluation_results/model_comparison.txt")
    
if __name__ == "__main__":
    main()