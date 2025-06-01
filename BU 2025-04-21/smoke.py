import re
import pandas as pd

# Read the smoke log file
with open('smoke.log', 'r') as f:
    log_lines = f.readlines()

# Prepare regex patterns
step_pattern = re.compile(r'--- Learn Step (\d+) ---')
reward_pattern = re.compile(r'Sampled Rewards: .*mean=([-\d\.]+)')
q_pattern = re.compile(r'Q\(s,a\): .*mean=([-\d\.]+)')
loss_pattern = re.compile(r'Loss = ([\d\.]+)')
grad_pattern = re.compile(r'Grad Norm = ([\d\.]+)')

# Parse the log
data = []
current_step = None

for line in log_lines:
    step_match = step_pattern.search(line)
    if step_match:
        current_step = int(step_match.group(1))
        continue
    if current_step is not None:
        reward_match = reward_pattern.search(line)
        if reward_match:
            mean_reward = float(reward_match.group(1))
            # Initialize entry
            data.append({'step': current_step, 'mean_reward': mean_reward})
            continue
        q_match = q_pattern.search(line)
        if q_match and data and data[-1]['step'] == current_step:
            data[-1]['avg_q'] = float(q_match.group(1))
            continue
        loss_match = loss_pattern.search(line)
        if loss_match and data and data[-1]['step'] == current_step:
            data[-1]['loss'] = float(loss_match.group(1))
            continue
        grad_match = grad_pattern.search(line)
        if grad_match and data and data[-1]['step'] == current_step:
            data[-1]['grad_norm'] = float(grad_match.group(1))
            # After capturing grad norm, reset current_step
            current_step = None
            continue

# Create DataFrame
df = pd.DataFrame(data)

# Display the DataFrame
print("\nTraining Metrics:")
print(df)
