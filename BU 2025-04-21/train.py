
"""
Training utilities for the DQN Tetris agent.
This module contains the main training loop and evaluation functions.
Optimized for GPU training.
"""
import os
import time
import numpy as np
import torch
import gc
import signal
import sys
import random
from simple_tetris_env import SimpleTetrisEnv, TETROMINO_SHAPES
from agent import DQNAgent
from config import CONFIG
from agent import get_process_memory


def preprocess_state(state, binary=False, dtype=np.float32, include_piece_info=True):
    """
    Process observation into tensor format for DQN with improved state representation.
    
    Args:
        state: Observation from environment
        binary: Whether to use binary state representation
        dtype: Data type for the output
        include_piece_info: Whether to include additional channels for piece information
        
    Returns:
        Processed tensor in format [C, H, W] with optional additional channels
    """
    # If include_piece_info is False or the state is not a dictionary, use the original method
    if not include_piece_info or not isinstance(state, dict):
        # Handle standard grid data
        if isinstance(state, dict) and 'grid' in state:
            grid = state['grid']
        else:
            grid = state

        # Add channel dimension if it's 2D
        if len(grid.shape) == 2:
            grid = grid.reshape(grid.shape[0], grid.shape[1], 1)

        # Apply binary conversion if requested
        if binary:
            grid = (grid > 0).astype(dtype)

        # Transpose to [C, H, W] format expected by CNN
        grid = np.transpose(grid, (2, 0, 1)).astype(dtype)

        return grid

    # Enhanced preprocessing with piece information
    # Extract the grid and piece information
    grid = state['grid']
    current_piece = state.get('current_piece')
    piece_x = state.get('piece_x', 0)
    piece_y = state.get('piece_y', 0)
    piece_rotation = state.get('piece_rotation', 0)
    next_piece = state.get('next_piece')

    # Add channel dimension if it's 2D
    if len(grid.shape) == 2:
        grid = grid.reshape(grid.shape[0], grid.shape[1], 1)

    # Apply binary conversion if requested
    if binary:
        grid = (grid > 0).astype(dtype)

    # Get grid dimensions
    height, width = grid.shape[0], grid.shape[1]

    # Channel 1: Grid state (possibly multi-channel already)
    grid_channel = np.transpose(grid, (2, 0, 1)).astype(dtype)

    # Channel 2: Current piece position (IMPROVED - use actual piece shape)
    piece_pos_channel = np.zeros((1, height, width), dtype=dtype)

    # Render the actual current piece shape at its current position
    if current_piece is not None and piece_rotation is not None:
        # Get the current piece shape based on type and rotation
        if 0 <= current_piece < len(TETROMINO_SHAPES) and 0 <= piece_rotation < len(TETROMINO_SHAPES[current_piece]):
            piece_shape = TETROMINO_SHAPES[current_piece][piece_rotation]

            # Map the piece onto the grid at its current position
            for y in range(len(piece_shape)):
                for x in range(len(piece_shape[y])):
                    if piece_shape[y][x] == 1:
                        grid_y = piece_y + y
                        grid_x = piece_x + x
                        if 0 <= grid_y < height and 0 <= grid_x < width:
                            piece_pos_channel[0, grid_y, grid_x] = 1.0

    # Channel 3: Next piece preview (IMPROVED - show actual shape)
    next_piece_channel = np.zeros((1, height, width), dtype=dtype)

    if next_piece is not None and 0 <= next_piece < len(TETROMINO_SHAPES):
        # Get the shape of the next piece (first rotation)
        next_shape = TETROMINO_SHAPES[next_piece][0]

        # Position it at the top middle of the grid for preview
        preview_x = width // 2 - len(next_shape[0]) // 2
        preview_y = 1  # Near the top

        # Map the next piece shape onto the grid
        for y in range(len(next_shape)):
            for x in range(len(next_shape[y])):
                if next_shape[y][x] == 1:
                    grid_y = preview_y + y
                    grid_x = preview_x + x
                    if 0 <= grid_y < height and 0 <= grid_x < width:
                        # Use half intensity to differentiate
                        next_piece_channel[0, grid_y, grid_x] = 0.5

    # Channel 4: Rotation encoding (simplified to a single value)
    rotation_channel = np.zeros((1, height, width), dtype=dtype)
    if piece_rotation is not None:
        # Use a simple scalar representation in corner
        # Normalize to 0-1 range (max rotation is 3)
        rotation_channel[0, 0, 0] = piece_rotation / 3.0

    # Combine all channels (reduced from previous implementation)
    combined = np.vstack([
        grid_channel,
        piece_pos_channel,
        next_piece_channel,
        rotation_channel
    ])

    return combined

def train(env, agent, config=None, render=False, logger=None):
    """
    GPU-optimized training function for DQN agent with curriculum learning.
    """
    if config is None:
        config = CONFIG.copy()

    # Use provided logger or print directly
    log = logger.info if logger else print

    # Check for CUDA
    device = config.get("device", "cpu")
    if device == "cuda" and torch.cuda.is_available():
        log(f"Training on GPU: {torch.cuda.get_device_name(0)}")
        # Set PyTorch to use current CUDA device by default
        torch.cuda.set_device(0)
        # CUDA settings for better performance
        torch.backends.cudnn.benchmark = True
    else:
        if device == "cuda":
            log("CUDA requested but not available. Falling back to CPU.")
            config["device"] = "cpu"
        log("Training on CPU")
        # Set up PyTorch for CPU efficiency
        torch.set_num_threads(config.get("num_threads", 4))
        log(f"PyTorch using {torch.get_num_threads()} threads")

    # Setup checkpointing
    os.makedirs(config.get("checkpoint_dir", "checkpoints"), exist_ok=True)
    latest_checkpoint_path = os.path.join(
        config.get("checkpoint_dir", "checkpoints"),
        "checkpoint_latest.pt"
    )

    # Training metrics
    episode_rewards = []
    best_eval_reward = float('-inf')
    best_eval_lines = 0

    # Print initial setup
    log(f"Starting training with model type: {config.get('model_type', 'dqn')}")
    log(f"Batch size: {config.get('batch_size', 32)}, Learning rate: {config.get('learning_rate', 0.0005)}")

    # Memory usage report
    if device == "cuda" and torch.cuda.is_available():
        log(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024 / 1024:.1f} MB")
        log(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024 / 1024:.1f} MB")
    else:
        log(f"CPU memory usage: {get_process_memory():.1f} MB")

    # Check if using curriculum learning
    using_curriculum = config.get("use_curriculum_learning", True)
    total_episodes = config.get("num_episodes", 10000)

    # Whether to use enhanced preprocessing with piece information
    use_enhanced_preprocessing = config.get("use_enhanced_preprocessing", True)

    if using_curriculum:
        log("\n===== USING CURRICULUM LEARNING =====")
        log("Phase 1 (0-33%): Focus on discovering line clearing")
        log("Phase 2 (33-66%): Start teaching board organization")
        log("Phase 3 (66-100%): Fine-tune for optimal play")

    if use_enhanced_preprocessing:
        log("\n===== USING ENHANCED STATE PREPROCESSING =====")
        log("Including piece information in state representation")
        log("Using additional channels for current piece, next piece, and rotation")

    # Track progress time
    start_time = time.time()

    # Main training loop
    for episode in range(1, config.get("num_episodes", 10000) + 1):
        # Apply curriculum learning if enabled
        if using_curriculum:
            from config import get_curriculum_config
            curr_config = get_curriculum_config(
                config, episode, total_episodes)

            # Print phase transition
            if episode == 1 or episode == int(total_episodes * 0.33) + 1 or episode == int(total_episodes * 0.66) + 1:
                phase = "1" if episode <= int(
                    total_episodes * 0.33) else "2" if episode <= int(total_episodes * 0.66) else "3"
                log(f"\n===== ENTERING PHASE {phase} at episode {episode} =====")
                log(f"Height penalty: {curr_config['reward_height_weight']}")
                log(f"Holes penalty: {curr_config['reward_holes_weight']}")
                log(
                    f"Bumpiness penalty: {curr_config['reward_bumpiness_weight']}")
                log(
                    f"Line clearing reward: {curr_config['reward_lines_cleared_weight']}")
                log(
                    f"Potential lines reward: {curr_config['reward_potential_lines']}\n")
        else:
            curr_config = config

        state = env.reset()
        use_binary = curr_config.get("use_binary_states", False)

        # Use the enhanced preprocessing if enabled
        state = preprocess_state(
            state, binary=use_binary, include_piece_info=use_enhanced_preprocessing)
        episode_reward = 0
        loss_values = []
        step = 0
        episode_lines = 0
        potential_lines_max = 0

        # Episode start time
        episode_start_time = time.time()
        done = False

        while not done and step < curr_config.get("max_steps_per_episode", 2000):
            # Check for timeout
            if time.time() - episode_start_time > curr_config.get("episode_timeout", 60):
                log(f"Episode {episode} timed out after {step} steps.")
                break

            # Select action
            action = agent.select_action(state)

            # Take action
            next_state, reward, done, info = env.step(action)

            # Handle terminal state - don't preprocess next_state if done
            if done:
                next_state_processed = None
            else:
                # Use the enhanced preprocessing if enabled
                next_state_processed = preprocess_state(
                    next_state, binary=use_binary, include_piece_info=use_enhanced_preprocessing)

            # Update episode statistics
            step_lines_cleared = info.get('lines_cleared', 0)
            if step_lines_cleared > 0:
                log(
                    f"!!! TRAIN.PY: Detected step_lines_cleared={step_lines_cleared} from info dict !!!")
            episode_lines += step_lines_cleared
            potential_lines_max = max(
                potential_lines_max, info.get('potential_lines', 0))

            # Store transition and learn - now with properly handled terminal state
            agent.store_transition(
                state, action, next_state_processed, reward, done, info)

            # Only update weights every few steps for CPU efficiency
            # For GPU, we can update every step for better convergence
            if step % curr_config.get("update_frequency", 1) == 0:
                loss = agent.learn()
                if loss is not None:
                    loss_values.append(loss)

            # Update state and metrics
            if not done:  # Only update state if not done
                state = next_state_processed
            episode_reward += reward
            step += 1

            # Render if requested
            if render:
                env.render()

        # Update epsilon for exploration
        agent.update_epsilon()

        # Update target network if using hard updates
        if not curr_config.get("use_soft_update", False) and episode % curr_config.get("target_update", 10) == 0:
            agent.update_target_network()

        # Track rewards
        episode_rewards.append(episode_reward)
        agent.add_episode_reward(episode_reward)

        # Print progress
        if episode % 5 == 0:
            avg_loss = np.mean(loss_values) if loss_values else float('nan')
            recent_rewards = episode_rewards[-10:] if len(
                episode_rewards) >= 10 else episode_rewards
            avg_reward = sum(recent_rewards) / len(recent_rewards)

            elapsed_time = time.time() - start_time
            episodes_per_hour = episode / (elapsed_time / 3600)
            estimated_total_hours = total_episodes / episodes_per_hour
            estimated_remaining_hours = estimated_total_hours - \
                (elapsed_time / 3600)

            log(f"Episode {episode}/{config.get('num_episodes', 10000)}, "
                f"Reward: {episode_reward:.2f}, "
                f"Avg(10): {avg_reward:.2f}, "
                f"Loss: {avg_loss:.5f}, "
                f"Lines: {episode_lines}, "
                f"Steps: {step}")

            # Add time estimates
            log(f"Time elapsed: {elapsed_time/3600:.1f}h, Est. remaining: {estimated_remaining_hours:.1f}h, "
                f"Episodes/hour: {episodes_per_hour:.1f}")

            # Memory usage report
            if curr_config.get("device", "cpu") == "cuda" and torch.cuda.is_available():
                log(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024 / 1024:.1f} MB, "
                    f"reserved: {torch.cuda.memory_reserved() / 1024 / 1024:.1f} MB, "
                    f"Epsilon: {agent.epsilon:.4f}")
            else:
                log(f"Memory usage: {get_process_memory():.1f} MB, Epsilon: {agent.epsilon:.4f}")

        # Garbage collection (less frequent for GPU)
        if episode % curr_config.get("gc_frequency", 50) == 0:
            gc.collect()
            if curr_config.get("device", "cpu") == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Checkpointing (more frequent for reliability against preemption)
        if episode % curr_config.get("checkpoint_frequency", 25) == 0:
            agent.save(latest_checkpoint_path)
            log(f"Checkpoint saved at episode {episode}")

            # Create numbered checkpoint less frequently
            if episode % (curr_config.get("checkpoint_frequency", 25) * 2) == 0:
                checkpoint_path = os.path.join(
                    curr_config.get("checkpoint_dir", "checkpoints"),
                    f"checkpoint_episode_{episode}.pt"
                )
                agent.save(checkpoint_path)

        # Evaluation
        if episode % curr_config.get("eval_frequency", 50) == 0:
            log(f"\nEvaluating agent at episode {episode}...")

            # Create a separate env with EXACTLY THE SAME GRID DIMENSIONS
            grid_width = env.grid_width
            grid_height = env.grid_height
            eval_env = SimpleTetrisEnv(
                grid_width=grid_width, grid_height=grid_height)

            # Call the external evaluate function with the same grid dimensions
            avg_eval_reward, total_eval_lines = evaluate(
                agent, eval_env,
                num_episodes=curr_config.get("eval_episodes", 3),
                render=render,
                use_enhanced_preprocessing=use_enhanced_preprocessing
            )

            # Save best model - prioritize models that clear lines
            is_best = False

            # Update best model criteria: clear lines OR get better reward
            if total_eval_lines > best_eval_lines:
                is_best = True
                best_eval_lines = total_eval_lines
                best_eval_reward = max(best_eval_reward, avg_eval_reward)
                log(f"New best model with {best_eval_lines} lines cleared!")
            elif total_eval_lines == best_eval_lines and avg_eval_reward > best_eval_reward:
                is_best = True
                best_eval_reward = avg_eval_reward
                log(
                    f"New best model with same lines ({best_eval_lines}) but better reward: {best_eval_reward:.2f}")

            if is_best:
                best_model_path = os.path.join(
                    curr_config.get("model_dir", "models"),
                    "best_model.pt"
                )
                agent.save(best_model_path)
                log(
                    f"New best model saved with reward: {best_eval_reward:.2f} and lines: {best_eval_lines}")

            # Clean up evaluation environment
            eval_env.close()
            log("Evaluation complete, continuing training...\n")

    # Final evaluation
    log("\nRunning final evaluation...")
    eval_env = SimpleTetrisEnv()

    # Call the external evaluate function
    avg_final_reward, total_lines = evaluate(agent, eval_env, num_episodes=config["eval_episodes"],
                                             render=render, use_enhanced_preprocessing=use_enhanced_preprocessing)

    # Save final model
    final_model_path = os.path.join(config.get(
        "model_dir", "models"), "final_model.pt")
    agent.save(final_model_path)
    log(f"Final model saved to {final_model_path}")

    # Clean up evaluation environment
    eval_env.close()

    # Print final training statistics
    log("\n===== TRAINING COMPLETE =====")
    log(f"Total episodes: {config.get('num_episodes', 10000)}")
    log(f"Best evaluation reward: {best_eval_reward:.2f}")
    log(f"Best lines cleared: {best_eval_lines}")
    log(f"Final evaluation reward: {avg_final_reward:.2f}")
    log(f"Final lines cleared: {total_lines}")
    log(f"Total training time: {(time.time() - start_time) / 3600:.2f} hours")

    return agent

def evaluate(agent, env, num_episodes=5, render=False, use_enhanced_preprocessing=True, logger=None):
    """
    Evaluate the agent's performance without artificial step limits.
    
    Args:
        agent: DQN agent to evaluate
        env: Environment to evaluate in
        num_episodes: Number of episodes to evaluate
        render: Whether to render the environment
        use_enhanced_preprocessing: Whether to use enhanced state preprocessing
        logger: Optional logger instance for output
        
    Returns:
        Tuple of (average reward, total lines cleared) across episodes
    """
    # Use provided logger or print directly
    log = logger.info if logger else print

    eval_rewards = []
    eval_lines = []
    eval_steps = []
    use_binary = CONFIG.get("use_binary_states", False)

    log("Starting evaluation - running until game over (no step limit)...")

    for episode in range(num_episodes):
        state = env.reset()

        # Save the raw state for debugging
        raw_state = state

        # Process the state
        state = preprocess_state(
            state, binary=use_binary, include_piece_info=use_enhanced_preprocessing)

        # Initialize tracking variables
        episode_reward = 0
        lines_cleared = 0
        steps = 0
        done = False

        # Track actions for debugging purposes
        action_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

        # Detect consecutive identical actions as a sign of being stuck
        last_actions = []
        max_repeats = 50

        log(f"Starting evaluation episode {episode+1}/{num_episodes}")

        # Safety limits to prevent infinite loops
        safety_limit = CONFIG.get("eval_safety_limit", 5000)
        last_lines_step = 0
        no_progress_limit = CONFIG.get("eval_early_stop_no_lines", 3000)
        # Give agent more time before checking progress
        min_steps_for_progress_check = 200

        # Maximum board height before forced termination
        max_board_height = env.grid_height - 2

        # Previous board state for detecting stuck conditions
        prev_board_state = None
        identical_board_count = 0
        max_identical_boards = 50

        # For debugging: track potential line clear opportunities
        potential_lines = 0

        while not done and steps < safety_limit:
            steps += 1

            if render:
                env.render()
                time.sleep(0.01)  # Small delay for visualization

            # Select action with small chance of exploration
            if random.random() < 0.05:  # 5% exploration during evaluation
                action = random.randint(0, env.action_space.n - 1)
            else:
                action = agent.select_action(state, training=False)

            # Update action count for analysis
            action_counts[action] += 1

            # Track repeated actions
            last_actions.append(action)
            if len(last_actions) > max_repeats:
                last_actions.pop(0)
                if len(set(last_actions)) == 1 and steps > 100:
                    log(
                        f"  Agent stuck in action loop (repeating action {action}). Forcing termination.")
                    done = True
                    break

            # Take action
            next_state, reward, done, info = env.step(action)

            # IMPORTANT: Double-check for game over condition from environment
            if hasattr(env, 'game_over') and env.game_over:
                done = True
                log(f"  Detected explicit game_over state at step {steps}")

            # MANUAL GAME OVER CHECK: Check if any pieces at the top of the board
            if isinstance(next_state, dict) and 'grid' in next_state:
                grid = next_state['grid']
                # Check top 2 rows for any filled cells
                if np.any(grid[0:2, :] > 0):
                    log(
                        f"  Detected filled cells at top of board. Forcing game over at step {steps}")
                    done = True

            # Check for board height exceeding threshold
            board_height = info.get('board_height', 0)
            if board_height >= max_board_height:
                log(f"  Board height ({board_height}) exceeds safety threshold. Forcing termination.")
                done = True

            # Check for unchanging board state (stuck condition)
            if isinstance(next_state, dict) and 'grid' in next_state:
                current_board_str = str(next_state['grid'])
                if current_board_str == prev_board_state and action == 5:  # NOTHING action
                    identical_board_count += 1
                    if identical_board_count >= max_identical_boards:
                        log(
                            f"  Board state unchanged for {identical_board_count} steps. Forcing termination.")
                        done = True
                else:
                    identical_board_count = 0
                    prev_board_state = current_board_str

            # Process next state (with correct terminal state handling)
            if done:
                next_state_processed = None
            else:
                next_state_processed = preprocess_state(
                    next_state, binary=use_binary, include_piece_info=use_enhanced_preprocessing)

            # Update state and metrics
            if not done:
                state = next_state_processed
            episode_reward += reward

            # Track lines cleared for progress detection
            new_lines = info.get('lines_cleared', 0)
            if new_lines > 0:
                lines_cleared += new_lines
                last_lines_step = steps  # Reset the counter when lines are cleared
                log(
                    f"  !!! LINE CLEARED at step {steps} - agent cleared {new_lines} lines!")

            # Track potential lines for debugging
            potential = info.get('potential_lines', 0)
            if potential > potential_lines:
                potential_lines = potential

            # Check for no progress (no lines cleared for too long) - only after minimum steps
            if no_progress_limit > 0 and steps > min_steps_for_progress_check and steps - last_lines_step > no_progress_limit:
                log(
                    f"  No lines cleared for {steps - last_lines_step} steps. Ending evaluation.")
                break

            # Print progress occasionally
            if steps % 100 == 0:
                log(f"  Episode {episode+1}, Step {steps}, Current reward: {episode_reward:.2f}, Lines: {lines_cleared}")
                log(f"  Board height: {board_height}, Holes: {info.get('holes', 0)}, Potential lines: {potential_lines}")

        # Report reason for episode completion
        if steps >= safety_limit:
            log(f"  Warning: Episode {episode+1} reached safety limit of {safety_limit} steps")
        elif steps - last_lines_step > no_progress_limit and steps > min_steps_for_progress_check:
            log(f"  Episode {episode+1} ended due to no progress for {steps - last_lines_step} steps")
        elif done:
            log(f"  Episode {episode+1} ended with game over")

        # Print action distribution for analysis
        log(f"  Action distribution: Left: {action_counts[0]}, Right: {action_counts[1]}, Rotate: {action_counts[2]}, Down: {action_counts[3]}, Drop: {action_counts[4]}, Nothing: {action_counts[5]}")
        log(f"  Episode {episode+1} complete: Reward={episode_reward:.2f}, Lines={lines_cleared}, Steps={steps}")

        eval_rewards.append(episode_reward)
        eval_lines.append(lines_cleared)
        eval_steps.append(steps)

    avg_reward = sum(eval_rewards) / len(eval_rewards)
    avg_lines = sum(eval_lines) / len(eval_lines)
    avg_steps = sum(eval_steps) / len(eval_steps)

    log(f"Evaluation complete:")
    log(f"  Average Reward: {avg_reward:.2f}")
    log(f"  Average Lines: {avg_lines:.2f}")
    log(f"  Average Steps: {avg_steps:.2f}")

    # Return both reward and total lines cleared
    return avg_reward, sum(eval_lines)

def main():
    """Main function for standalone running."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train DQN Tetris agent")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--episodes", type=int, default=CONFIG.get("num_episodes", 1000), 
                        help="Number of episodes to train")
    parser.add_argument("--render", action="store_true", help="Render environment during training")
    parser.add_argument("--disable-enhanced", action="store_true", 
                        help="Disable enhanced preprocessing (piece information)")
    args = parser.parse_args()
    
    # Update CONFIG
    CONFIG["num_episodes"] = args.episodes
    CONFIG["use_enhanced_preprocessing"] = not args.disable_enhanced
    
    # Set render mode based on args
    render_mode = "human" if args.render else None
    
    # Create environment
    env = SimpleTetrisEnv(render_mode=render_mode)
    
    # Check for resuming
    if args.resume:
        latest_checkpoint_path = os.path.join(CONFIG.get("checkpoint_dir", "checkpoints"), "checkpoint_latest.pt")
        if os.path.exists(latest_checkpoint_path):
            print(f"Resuming from checkpoint: {latest_checkpoint_path}")
            
            # Create a temporary agent to get the state shape
            temp_state = env.reset()
            temp_state = preprocess_state(temp_state, include_piece_info=CONFIG["use_enhanced_preprocessing"])
            input_shape = temp_state.shape
            
            # Create agent 
            agent = DQNAgent(
                input_shape=input_shape,
                n_actions=env.action_space.n,
                device=CONFIG.get("device", "cpu"),
                config=CONFIG
            )
            
            # Load checkpoint
            agent.load(latest_checkpoint_path)
            
            # Update episodes to include already trained episodes
            CONFIG["num_episodes"] = args.episodes + agent.episode_count
            print(f"Continuing training from episode {agent.episode_count} to {CONFIG['num_episodes']}")
        else:
            print("No checkpoint found. Starting fresh training.")
            agent = None
    else:
        agent = None
    
    # Create agent if not loaded from checkpoint
    if agent is None:
        state = env.reset()
        state = preprocess_state(state, binary=CONFIG.get("use_binary_states", False), 
                              include_piece_info=CONFIG["use_enhanced_preprocessing"])
        input_shape = state.shape
        
        agent = DQNAgent(
            input_shape=input_shape,
            n_actions=env.action_space.n,
            device=CONFIG.get("device", "cpu"),
            config=CONFIG
        )
    
    # Train the agent
    train(env, agent, config=CONFIG, render=args.render)
    
    # Close environment
    env.close()

if __name__ == "__main__":
    main()