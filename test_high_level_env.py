import numpy as np
from high_level_env import HighLevelTetrisEnv, visualize_placements
from simple_tetris_env import SimpleTetrisEnv
import time

def test_high_level_actions():
    """Test the high-level action wrapper."""
    print("Testing High-Level Tetris Environment")
    
    # Create environment
    env = HighLevelTetrisEnv()
    
    # Reset and check placements
    state = env.reset()
    print(f"Number of valid placements: {len(env.valid_placements)}")
    
    # Take a random action
    action = np.random.randint(0, len(env.valid_placements))
    placement = env.valid_placements[action]
    print(f"Selected placement: Rotation={placement[0]}, X={placement[1]}, Y={placement[2]}")
    
    # Execute the action
    next_state, reward, done, info = env.step(action)
    print(f"Result: reward={reward}, done={done}")
    
    # Visual test
    if not done:
        print("Visualizing possible placements for the next piece...")
        viz_grids = visualize_placements(env, top_k=3)
        for i, grid in enumerate(viz_grids):
            print(f"Placement option {i+1}:")
            for row in grid:
                print("".join(str(cell) if cell > 0 else "." for cell in row))
    
    print("High-level environment test complete!")

def compare_performance():
    """Compare performance between low-level and high-level environments."""
    print("Comparing Low-Level vs High-Level Environment Performance")
    
    # Parameters
    num_episodes = 10
    max_steps = 1000
    
    # Test low-level environment
    env_low = SimpleTetrisEnv()
    total_reward_low = 0
    total_lines_low = 0
    start_time_low = time.time()
    
    for ep in range(num_episodes):
        state = env_low.reset()
        episode_reward = 0
        episode_lines = 0
        
        for step in range(max_steps):
            action = np.random.randint(0, 6)  # Random action
            next_state, reward, done, info = env_low.step(action)
            episode_reward += reward
            episode_lines += info.get('lines_cleared', 0)
            
            if done:
                break
        
        total_reward_low += episode_reward
        total_lines_low += episode_lines
        print(f"Low-Level Episode {ep+1}: Reward={episode_reward:.2f}, Lines={episode_lines}")
    
    time_low = time.time() - start_time_low
    
    # Test high-level environment
    env_high = HighLevelTetrisEnv()
    total_reward_high = 0
    total_lines_high = 0
    start_time_high = time.time()
    
    for ep in range(num_episodes):
        state = env_high.reset()
        episode_reward = 0
        episode_lines = 0
        
        for step in range(max_steps):
            # Random high-level action
            action = np.random.randint(0, len(env_high.valid_placements))
            next_state, reward, done, info = env_high.step(action)
            episode_reward += reward
            episode_lines += info.get('lines_cleared', 0)
            
            if done:
                break
        
        total_reward_high += episode_reward
        total_lines_high += episode_lines
        print(f"High-Level Episode {ep+1}: Reward={episode_reward:.2f}, Lines={episode_lines}")
    
    time_high = time.time() - start_time_high
    
    # Compare results
    print("\nPerformance Comparison:")
    print(f"Low-Level: Avg Reward={total_reward_low/num_episodes:.2f}, Avg Lines={total_lines_low/num_episodes:.2f}, Time={time_low:.2f}s")
    print(f"High-Level: Avg Reward={total_reward_high/num_episodes:.2f}, Avg Lines={total_lines_high/num_episodes:.2f}, Time={time_high:.2f}s")
    print(f"Speed Difference: {time_low/time_high:.2f}x")

if __name__ == "__main__":
    test_high_level_actions()
    print("\n" + "="*50 + "\n")
    compare_performance()