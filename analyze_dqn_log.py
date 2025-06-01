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
    min_length = min(len(data[key]) for key in data.keys() if data[key]) if any(data.values()) else 0
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

def extract_line_clearing_data(log_content):
    """Extract data about line clearing events, including inference from high rewards."""
    # Look for both specific line clearing events and debug printouts
    line_clear_pattern = r"LINE CLEARED at step (\d+) - agent cleared (\d+) lines"
    debug_line_clear_pattern = r"DEBUG: Detected (\d+) completed lines at step (\d+)"
    line_clear_reward_pattern = r"@@@ LINE CLEAR REWARD: Base=([\d\.]+) for (\d+) lines @@@"
    high_reward_pattern = r"Episode (\d+)/\d+, Steps: (\d+), Reward: ([89]\d\d\.\d+|1\d\d\d\.\d+), Avg"
    
    line_clearing_events = re.findall(line_clear_pattern, log_content)
    debug_events = re.findall(debug_line_clear_pattern, log_content)
    reward_events = re.findall(line_clear_reward_pattern, log_content)
    high_reward_events = re.findall(high_reward_pattern, log_content)
    
    # Convert to structured data
    events = []
    
    # Process regular line clearing events
    for step, lines in line_clearing_events:
        events.append({
            'step': int(step),
            'lines': int(lines),
            'source': 'evaluation'
        })
    
    # Process debug events
    for lines, step in debug_events:
        events.append({
            'step': int(step),
            'lines': int(lines),
            'source': 'debug'
        })
    
    # Process reward events (these happen during training)
    for reward, lines in reward_events:
        events.append({
            'reward': float(reward),
            'lines': int(lines),
            'source': 'reward'
        })
    
    # Process high reward events (inferred line clears)
    for episode, step, reward in high_reward_events:
        # Assuming rewards in 800-999 range are from single/double line clears
        # and rewards 1000+ are from triple/tetris clears
        reward_value = float(reward)
        inferred_lines = 1  # Default assumption
        
        if reward_value >= 1500:
            inferred_lines = 4  # Likely a Tetris (4 lines)
        elif reward_value >= 1000:
            inferred_lines = 3  # Likely a triple
        elif reward_value >= 900:
            inferred_lines = 2  # Likely a double
        
        events.append({
            'episode': int(episode),
            'step': int(step),
            'reward': reward_value,
            'lines': inferred_lines,
            'source': 'inferred'
        })
    
    # Add to analysis
    line_clear_data = {
        'total_events': len(events),
        'total_lines': sum(event.get('lines', 0) for event in events),
        'events': events,
        'explicit_events': len(line_clearing_events) + len(debug_events) + len(reward_events),
        'inferred_events': len(high_reward_events)
    }
    
    return line_clear_data

def extract_action_distribution(log_content):
    """Extract data about action distribution."""
    action_pattern = r"Action distribution: Left: (\d+), Right: (\d+), Rotate: (\d+), Down: (\d+), Drop: (\d+), Nothing: (\d+)"
    action_matches = re.findall(action_pattern, log_content)
    
    if not action_matches:
        return None
        
    # Convert to structured data
    actions = []
    for match in action_matches:
        left, right, rotate, down, drop, nothing = map(int, match)
        actions.append({
            'left': left,
            'right': right,
            'rotate': rotate,
            'down': down,
            'drop': drop,
            'nothing': nothing,
            'total': left + right + rotate + down + drop + nothing
        })
    
    # Calculate average distribution
    if actions:
        avg_dist = {
            'left': np.mean([a['left'] for a in actions]),
            'right': np.mean([a['right'] for a in actions]),
            'rotate': np.mean([a['rotate'] for a in actions]),
            'down': np.mean([a['down'] for a in actions]),
            'drop': np.mean([a['drop'] for a in actions]),
            'nothing': np.mean([a['nothing'] for a in actions])
        }
    else:
        avg_dist = None
    
    return {
        'actions': actions,
        'avg_dist': avg_dist
    }

def extract_board_state_info(log_content):
    """Extract information about board states."""
    # Look for patterns that might show board state information
    height_pattern = r"Board height: (\d+), Holes: (\d+)"
    potential_pattern = r"Potential lines: ([\d\.]+)"
    
    height_matches = re.findall(height_pattern, log_content)
    potential_matches = re.findall(potential_pattern, log_content)
    
    data = {
        'height_data': [],
        'potential_data': []
    }
    
    # Process height and holes data
    for height, holes in height_matches:
        data['height_data'].append({
            'height': int(height),
            'holes': int(holes)
        })
    
    # Process potential lines data
    for potential in potential_matches:
        data['potential_data'].append(float(potential))
    
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
    
    # Extract reward structure
    reward_patterns = {
        'height_weight': r"reward_height_weight[\"']*: ([-\d\.]+)",
        'holes_weight': r"reward_holes_weight[\"']*: ([-\d\.]+)",
        'lines_cleared_weight': r"reward_lines_cleared_weight[\"']*: ([\d\.]+)",
        'bumpiness_weight': r"reward_bumpiness_weight[\"']*: ([-\d\.]+)",
        'tetris_bonus': r"reward_tetris_bonus[\"']*: ([\d\.]+)",
        'game_over_penalty': r"reward_game_over_penalty[\"']*: ([-\d\.]+)",
    }
    
    for key, pattern in reward_patterns.items():
        match = re.search(pattern, log_content)
        if match:
            config[key] = float(match.group(1))
    
    return config

def extract_epsilon_data(log_content):
    """Extract epsilon (exploration rate) data."""
    epsilon_pattern = r"Current exploration rate \(epsilon\): ([\d\.]+)"
    epsilon_matches = re.findall(epsilon_pattern, log_content)
    
    epsilon_values = [float(eps) for eps in epsilon_matches]
    
    return epsilon_values

def analyze_training(learn_data, episode_data, config, line_clear_data, action_data, board_state_info, epsilon_data, log_content):
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
    
    # Reward structure summary
    analysis.append("\n=== REWARD STRUCTURE ===")
    if 'lines_cleared_weight' in config:
        analysis.append(f"Line Clearing Reward: {config['lines_cleared_weight']}")
    if 'tetris_bonus' in config:
        analysis.append(f"Tetris Bonus: {config['tetris_bonus']}")
    if 'height_weight' in config:
        analysis.append(f"Height Penalty: {config['height_weight']}")
    if 'holes_weight' in config:
        analysis.append(f"Holes Penalty: {config['holes_weight']}")
    if 'bumpiness_weight' in config:
        analysis.append(f"Bumpiness Penalty: {config['bumpiness_weight']}")
    if 'game_over_penalty' in config:
        analysis.append(f"Game Over Penalty: {config['game_over_penalty']}")
    
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
    
    # Exploration rate analysis
    if epsilon_data:
        analysis.append("\n=== EXPLORATION ANALYSIS ===")
        analysis.append(f"Latest epsilon: {epsilon_data[-1]:.4f}")
        analysis.append(f"Epsilon range: {min(epsilon_data):.4f} - {max(epsilon_data):.4f}")
        
        # Check if exploration rate is too low too early
        if len(episode_data['episode']) > 0 and max(episode_data['episode']) < 1000 and min(epsilon_data) < 0.3:
            analysis.append("WARNING: Exploration rate (epsilon) is decreasing too quickly. Agent may not explore enough.")
    
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
    
    # Line clearing analysis
    analysis.append("\n=== LINE CLEARING ANALYSIS ===")
    if line_clear_data and line_clear_data['total_events'] > 0:
        if line_clear_data['explicit_events'] > 0:
            analysis.append(f"Explicit line clearing events: {line_clear_data['explicit_events']}")
        
        if line_clear_data['inferred_events'] > 0:
            analysis.append(f"Inferred line clearing events from high rewards: {line_clear_data['inferred_events']}")
        
        analysis.append(f"Total estimated lines cleared: {line_clear_data['total_lines']}")
        
        if line_clear_data['inferred_events'] > 0 and line_clear_data['explicit_events'] == 0:
            analysis.append("\nNOTE: No explicit line clear logging found, but high rewards strongly suggest line clearing is occurring")
            analysis.append("Consider adding explicit logging with: print(f\"LINE CLEARED - {lines} lines at step {step}\")")
        
        # Calculate line clearing frequency
        if episode_data['episode'] and episode_data['episode'][-1] > 0:
            total_episodes = episode_data['episode'][-1]
            clearing_ratio = line_clear_data['inferred_events'] / total_episodes
            analysis.append(f"\nLine clearing efficiency: {clearing_ratio:.3f} clearing episodes per episode")
            analysis.append(f"Average lines per clearing episode: {line_clear_data['total_lines']/line_clear_data['inferred_events']:.2f}")
        
        # Performance trend
        if len(line_clear_data['events']) >= 10:
            early_events = line_clear_data['events'][:len(line_clear_data['events'])//2]
            late_events = line_clear_data['events'][len(line_clear_data['events'])//2:]
            
            early_avg_lines = sum(e.get('lines', 0) for e in early_events) / len(early_events)
            late_avg_lines = sum(e.get('lines', 0) for e in late_events) / len(late_events)
            
            if late_avg_lines > early_avg_lines:
                analysis.append(f"POSITIVE: Line clearing efficiency is improving (Early: {early_avg_lines:.2f}, Late: {late_avg_lines:.2f} lines per event)")
            else:
                analysis.append(f"Line clearing efficiency is stable or declining (Early: {early_avg_lines:.2f}, Late: {late_avg_lines:.2f} lines per event)")
    else:
        analysis.append("WARNING: No line clearing events detected. Agent may not be clearing lines.")
        analysis.append("Recommendations:")
        analysis.append("1. Add explicit line clear logging to confirm behavior")
        analysis.append("2. Check reward structure to ensure line clearing is properly incentivized")
    # Board state analysis
    if board_state_info and (board_state_info['height_data'] or board_state_info['potential_data']):
        analysis.append("\n=== BOARD STATE ANALYSIS ===")
        
        # Height and holes analysis
        if board_state_info['height_data']:
            heights = [d['height'] for d in board_state_info['height_data']]
            holes = [d['holes'] for d in board_state_info['height_data']]
            
            avg_height = np.mean(heights)
            avg_holes = np.mean(holes)
            max_height = max(heights)
            max_holes = max(holes)
            
            analysis.append(f"Average board height: {avg_height:.1f}")
            analysis.append(f"Maximum board height: {max_height}")
            analysis.append(f"Average holes: {avg_holes:.1f}")
            analysis.append(f"Maximum holes: {max_holes}")
            
            # Check if board is getting too high
            if avg_height > 10:  # Assuming grid height = 14
                analysis.append("WARNING: Average board height is high. Agent is struggling to clear lines.")
            
            # Check if too many holes
            if avg_holes > 5:
                analysis.append("WARNING: High number of holes. Agent is not managing piece placement well.")
        
        # Potential lines analysis
        if board_state_info['potential_data']:
            avg_potential = np.mean(board_state_info['potential_data'])
            max_potential = max(board_state_info['potential_data'])
            
            analysis.append(f"Average potential lines: {avg_potential:.2f}")
            analysis.append(f"Maximum potential lines: {max_potential:.2f}")
            
            # Check if agent is setting up line clears
            if max_potential > 3 and line_clear_data['total_events'] == 0:
                analysis.append("NOTE: Agent is creating potential line clears but not executing them.")
    
    # Action distribution analysis
    if action_data and action_data['avg_dist']:
        analysis.append("\n=== ACTION DISTRIBUTION ANALYSIS ===")
        avg = action_data['avg_dist']
        total = sum(avg.values())
        
        # Calculate percentages
        pct = {k: (v/total*100) for k, v in avg.items()}
        
        analysis.append(f"Left: {pct['left']:.1f}%, Right: {pct['right']:.1f}%, Rotate: {pct['rotate']:.1f}%")
        analysis.append(f"Down: {pct['down']:.1f}%, Drop: {pct['drop']:.1f}%, Nothing: {pct['nothing']:.1f}%")
        
        # Check for concerning patterns
        if pct['nothing'] > 50:
            analysis.append("WARNING: Agent is choosing 'do nothing' too frequently (>50%).")
        
        if pct['drop'] < 5:
            analysis.append("WARNING: Agent rarely uses 'drop' action (<5%). May not be effective at placing pieces.")
        
        if pct['rotate'] < 10:
            analysis.append("WARNING: Agent rarely rotates pieces (<10%). Not optimizing piece placement.")
    
    # Health check and recommendations
    analysis.append("\n=== TRAINING HEALTH CHECK ===")
    
    # Check for line clearing issues
    if line_clear_data['total_events'] == 0:
        analysis.append("CRITICAL: Agent is not clearing any lines. This is the most important issue to fix.")
    
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
    
    # Line clearing recommendations
    if line_clear_data['total_events'] == 0:
        analysis.append("1. CRITICAL: Increase line clearing rewards dramatically:")
        analysis.append("   - Set reward_lines_cleared_weight to 10000.0")
        analysis.append("   - Set reward_tetris_bonus to 20000.0")
        analysis.append("   - Consider reducing other rewards/penalties to make line clearing dominant")
    
    # Check if reward is improving
    if episode_data['reward'] and len(episode_data['reward']) > 5:
        if np.mean(episode_data['reward'][-5:]) < np.mean(episode_data['reward'][:5]):
            analysis.append("2. Agent is not improving. Consider revising reward function or exploring more.")
    
    # Check for potential exploration issues
    if epsilon_data and min(epsilon_data) < 0.3 and len(episode_data['episode']) < 1000:
        analysis.append("3. Exploration rate is decreasing too quickly. Slow down epsilon decay.")
    
    # Check for gradient issues
    if learn_data['grad_norm'] and np.mean(learn_data['grad_norm']) > 30:
        analysis.append("4. Gradient norms are high. Consider reducing learning rate or implementing gradient clipping.")
    
    # Check for value function issues
    if learn_data['q_mean'] and np.mean(learn_data['q_mean'][-min(20, len(learn_data['q_mean'])):]) < -20:
        analysis.append("5. Q-values are very negative. Consider scaling rewards or adjusting discount factor.")
    
    # Action distribution recommendations
    if action_data and action_data['avg_dist']:
        if action_data['avg_dist']['nothing'] > 0.5 * sum(action_data['avg_dist'].values()):
            analysis.append("6. Agent is using 'do nothing' action too often. Check reward structure.")
    
    # Curriculum learning recommendation
    if config.get('curriculum') == True and line_clear_data['total_events'] == 0:
        analysis.append("7. Consider disabling curriculum learning until agent learns basic line clearing.")
    
    return analysis

def generate_plots(learn_data, episode_data, line_clear_data, action_data, board_state_info, epsilon_data, output_prefix):
    """Generate plots of training metrics."""
    # Directory for plots
    plots_dir = f"{output_prefix}_plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot 1: Loss over time
    if learn_data['step'] and learn_data['loss']:
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
    if learn_data['step'] and (learn_data['q_mean'] or learn_data['avg_q']):
        plt.figure(figsize=(10, 6))
        if learn_data['q_mean']:
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
    if learn_data['step'] and learn_data['grad_norm']:
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
        if episode_data['avg_reward'] and len(episode_data['avg_reward']) == len(episode_data['episode']):
            plt.plot(episode_data['episode'], episode_data['avg_reward'], 
                     label=f'Moving Avg (window={episode_data["avg_window"][0] if episode_data["avg_window"] else "N/A"})')
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
        # Plot 5: Episode lengths (continued)
        plt.savefig(f"{plots_dir}/episode_lengths.png")
        plt.close()
    
    # Plot 6: Line clearing events
    if line_clear_data and line_clear_data['events']:
        plt.figure(figsize=(10, 6))
        events = line_clear_data['events']
        event_indices = list(range(len(events)))
        lines_cleared = [e['lines'] for e in events]
        plt.bar(event_indices, lines_cleared)
        plt.xlabel('Line Clear Event Index')
        plt.ylabel('Lines Cleared')
        plt.title('Lines Cleared per Event')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{plots_dir}/line_clearing.png")
        plt.close()
    
    # Plot 7: Action distribution
    if action_data and action_data['avg_dist']:
        plt.figure(figsize=(10, 6))
        avg = action_data['avg_dist']
        plt.bar(avg.keys(), avg.values())
        plt.xlabel('Action')
        plt.ylabel('Average Count')
        plt.title('Action Distribution')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{plots_dir}/action_distribution.png")
        plt.close()
    
    # Plot 8: Board height and holes
    if board_state_info and board_state_info['height_data']:
        plt.figure(figsize=(12, 6))
        heights = [d['height'] for d in board_state_info['height_data']]
        holes = [d['holes'] for d in board_state_info['height_data']]
        indices = list(range(len(heights)))
        
        plt.subplot(1, 2, 1)
        plt.plot(indices, heights)
        plt.xlabel('Observation Index')
        plt.ylabel('Board Height')
        plt.title('Board Height Over Time')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(indices, holes)
        plt.xlabel('Observation Index')
        plt.ylabel('Number of Holes')
        plt.title('Holes Over Time')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{plots_dir}/board_state.png")
        plt.close()
    
    # Plot 9: Epsilon (exploration rate) over time
    if epsilon_data:
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(epsilon_data)), epsilon_data)
        plt.xlabel('Observation Index')
        plt.ylabel('Epsilon')
        plt.title('Exploration Rate (Epsilon) Over Time')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{plots_dir}/epsilon.png")
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
    line_clear_data = extract_line_clearing_data(log_content)
    action_data = extract_action_distribution(log_content)
    board_state_info = extract_board_state_info(log_content)
    epsilon_data = extract_epsilon_data(log_content)
    
    # Analyze data
    analysis = analyze_training(learn_data, episode_data, config_data, 
                               line_clear_data, action_data, board_state_info,
                               epsilon_data, log_content)
    
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
        generate_plots(learn_data, episode_data, line_clear_data, 
                      action_data, board_state_info, epsilon_data, output_prefix)

if __name__ == '__main__':
    main()