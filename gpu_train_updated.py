import os
import time
import numpy as np
import torch
import gc
from agent import DQNAgent
from config import CONFIG

# Add this function near the top of gpu_train_updated.py, after the imports
def monitor_training_progress(agent, env, config, logger=None, placement_stats=None):
    log = logger.info if logger else print
    log("\n===== TRAINING PROGRESS REPORT =====")
    log(f"Global episodes: {agent.episode_count}")
    log(f"Training steps: {agent.training_steps}")
    log(f"Current epsilon: {agent.epsilon:.4f}")
    avg_reward = agent.get_average_reward()
    avg_q = agent.get_average_q_value()
    avg_loss = agent.get_average_loss()
    log(f"Average reward (last 100 episodes): {avg_reward:.2f}" if avg_reward is not None else "Average reward: N/A")
    log(f"Average Q-value (last 100 batches): {avg_q:.2f}"     if avg_q is not None     else "Average Q-value: N/A")
    log(f"Average loss (last 100 batches): {avg_loss:.4f}"      if avg_loss is not None  else "Average loss: N/A")
    if config["device"] == "cuda" and torch.cuda.is_available():
        log(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024 / 1024:.1f} MB")
        log(f"GPU memory reserved:  {torch.cuda.memory_reserved()  / 1024 / 1024:.1f} MB")
    
    # Report high-level action statistics if available
    if placement_stats and (placement_stats.get('rotations') or placement_stats.get('x_positions')):
        log("\n=== High-Level Action Statistics ===")
        
        # Report rotation usage
        if placement_stats.get('rotations'):
            log("Rotation Usage:")
            total_rotations = sum(placement_stats['rotations'].values())
            for rotation, count in sorted(placement_stats['rotations'].items()):
                percentage = (count / total_rotations) * 100 if total_rotations > 0 else 0
                log(f"  Rotation {rotation}: {count} times ({percentage:.1f}%)")
        
        # Report x position usage
        if placement_stats.get('x_positions'):
            log("X Position Usage:")
            total_positions = sum(placement_stats['x_positions'].values())
            for pos, count in sorted(placement_stats['x_positions'].items()):
                percentage = (count / total_positions) * 100 if total_positions > 0 else 0
                log(f"  Column {pos}: {count} times ({percentage:.1f}%)")
    
    log("====================================\n")


def train_gpu_optimized(env, agent, config=None, render=False, logger=None):
    """
    GPU-optimized training function for DQN agent with vectorized environments.
    Updated to support high-level actions and track placement statistics.
    """
    if config is None:
        from config import CONFIG
        config = CONFIG.copy()

    log = logger.info if logger else print

    os.makedirs(config.get("checkpoint_dir", "checkpoints"), exist_ok=True)
    latest_checkpoint = os.path.join(config["checkpoint_dir"], "checkpoints", "checkpoint_latest.pt")
    last_checkpoint_ep = 0

    log(f"Starting GPU-optimized training with model type: {config.get('model_type', 'dqn')}")
    # right after you have created the agent and before training loop:
    current_lr = agent.optimizer.param_groups[0]['lr']
    log(f"Batch size: {config.get('batch_size', 32)}, Learning rate (optimizer): {current_lr}")

    if hasattr(env, "num_envs"):
        log(f"Using vectorized envs: {env.num_envs}")

    # Check if we're using high-level actions
    using_high_level = False
    if hasattr(env, "envs") and hasattr(env.envs[0], "valid_placements"):
        using_high_level = True
        log("Detected high-level action environment")
    elif hasattr(env, "valid_placements"):
        using_high_level = True
        log("Detected high-level action environment")
    
    # Initialize placement statistics tracking for high-level actions
    placement_stats = {} if using_high_level else None

    using_curriculum = config.get("use_curriculum_learning", True)
    total_eps = config.get("num_episodes", 10000)
    start_time = time.time()

    # init env
    if hasattr(env, "num_envs"):
        states = env.reset()
        N = env.num_envs
    else:
        states = [env.reset()]
        N = 1

    step = 0
    done_eps = 0
    use_amp = config.get("use_amp", False) and hasattr(agent, "learn_amp")

    # per‑env trackers
    eps_rewards = [0.0]*N
    eps_done = [False]*N
    rewards_log = []

    while done_eps < total_eps:
        # select actions
        actions = [
            agent.select_action(s) if s is not None else 0
            for s in states
        ]

        # step
        if hasattr(env, "step"):
            next_s, rews, dones, infos = env.step(actions)
        else:
            ns, r, d, i = env.step(actions[0])
            next_s, rews, dones, infos = [ns], [r], [d], [i]

        # curriculum
        curr_cfg = __import__("config").get_curriculum_config(config, done_eps, total_eps) if using_curriculum else config

        # store + handle episode endings
        for i in range(N):
            agent.store_transition(states[i], actions[i], next_s[i], rews[i], dones[i], infos[i])
            eps_rewards[i] += rews[i]

            # Track high-level action statistics if available
            if using_high_level and 'placement' in infos[i]:
                placement = infos[i]['placement']
                if placement:
                    rotation, x, y = placement
                    # Track rotation usage
                    placement_stats.setdefault('rotations', {}).setdefault(rotation, 0)
                    placement_stats['rotations'][rotation] += 1
                    
                    # Track x position usage
                    placement_stats.setdefault('x_positions', {}).setdefault(x, 0)
                    placement_stats['x_positions'][x] += 1

            if dones[i] and not eps_done[i]:
                eps_done[i] = True
                done_eps += 1
                rewards_log.append(eps_rewards[i])
                agent.add_episode_reward(eps_rewards[i])

                # periodic summary
                if done_eps % 10 == 0:
                    avg10 = sum(rewards_log[-10:]) / min(10, len(rewards_log))
                    elapsed = time.time() - start_time
                    log(f"Episode {done_eps}/{total_eps}, Steps: {step}, Reward: {rewards_log[-1]:.2f}, Avg(10): {avg10:.2f}, Eps/hr: {done_eps/elapsed*3600:.1f}")

                # reset trackers
                eps_rewards[i] = 0.0
                eps_done[i] = False

        # reset envs
        if hasattr(env, "reset_if_done"):
            states = env.reset_if_done()
        else:
            for i in range(N):
                if dones[i]:
                    states[i] = env.reset() if i == 0 else None
                else:
                    states[i] = next_s[i]

        # learn
        if step % curr_cfg.get("update_frequency", 1) == 0 and len(agent.memory) >= agent.batch_size:
            log(f"--- Learn Step {agent.training_steps+1} ---")
            if use_amp and not config.get("debug", False):
                loss = agent.learn_amp()
            else:
                loss = agent.learn()
            log(f"Learn call completed, loss: {loss}")

        agent.update_epsilon()

        # target network update
        if curr_cfg.get("use_soft_update", False):
            agent.update_target_network()
        elif done_eps and done_eps % curr_cfg.get("target_update", 10) == 0:
            agent.update_target_network()

        # Monitor training progress periodically
        if done_eps and done_eps % 100 == 0 and done_eps > last_checkpoint_ep:
            monitor_training_progress(agent, env, config, logger, placement_stats)

        # checkpointing
        freq = curr_cfg.get("checkpoint_frequency", 50)
        if done_eps and done_eps % freq == 0 and done_eps > last_checkpoint_ep:
            agent.save(latest_checkpoint)
            log(f"Checkpoint saved at episode {done_eps}")
            last_checkpoint_ep = done_eps

        step += 1

    # final save & report
    log("\nTraining complete!")
    log(f"Total episodes: {done_eps}, Total steps: {step}, Time: {(time.time()-start_time)/3600:.2f}h")
    
    # Final high-level action statistics
    if using_high_level and placement_stats:
        log("\n=== Final High-Level Action Statistics ===")
        
        # Report rotation usage
        if placement_stats.get('rotations'):
            log("Rotation Usage:")
            total_rotations = sum(placement_stats['rotations'].values())
            for rotation, count in sorted(placement_stats['rotations'].items()):
                percentage = (count / total_rotations) * 100 if total_rotations > 0 else 0
                log(f"  Rotation {rotation}: {count} times ({percentage:.1f}%)")
        
        # Report x position usage
        if placement_stats.get('x_positions'):
            log("X Position Usage:")
            total_positions = sum(placement_stats['x_positions'].values())
            for pos, count in sorted(placement_stats['x_positions'].items()):
                percentage = (count / total_positions) * 100 if total_positions > 0 else 0
                log(f"  Column {pos}: {count} times ({percentage:.1f}%)")
    
    final_path = os.path.join(config["model_dir"], "final_model.pt")
    agent.save(final_path)
    log(f"Final model saved to {final_path}")

    return agent