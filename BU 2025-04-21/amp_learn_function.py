# Add these imports at the top of amp_learn_function.py
import torch
import torch.nn as nn
import numpy as np
from collections import namedtuple

# Define the same Transition namedtuple as in replay.py and gpu_replay_buffer.py
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done', 'info'))

def learn_amp(self):
    """Update the network weights using a batch of experiences with Automatic Mixed Precision."""
    # Check if we have enough samples
    if len(self.memory) < self.batch_size:
        return None
    
    try:
        self.training_steps += 1
        
        # Initialize GradScaler for mixed precision training
        if not hasattr(self, 'scaler'):
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Initialize variables to avoid UnboundLocalError
        has_direct_tensors = False
        has_prioritized = hasattr(self.memory, 'update_priorities')
        importance_weights = None
        indices = None
        transitions = None
        state_batch = None
        action_batch = None
        reward_batch = None
        non_final_mask = None
        non_final_next_states = None
        done_batch = None
        
        # Sample batch - handle different replay buffer types
        if hasattr(self.memory, 'sample_tensors') and callable(getattr(self.memory, 'sample_tensors')):
            # The sample_tensors method might return different numbers of values
            # based on whether it's prioritized or not
            try:
                # Try using sample_tensors, which should return tensors directly
                returned_values = self.memory.sample_tensors(self.batch_size)
                
                if isinstance(returned_values, tuple):
                    if len(returned_values) == 8:  # For prioritized replay
                        state_batch, action_batch, non_final_next_states, reward_batch, done_batch, non_final_mask, indices, importance_weights = returned_values
                        has_direct_tensors = True
                    elif len(returned_values) == 6:  # For regular replay
                        state_batch, action_batch, non_final_next_states, reward_batch, done_batch, non_final_mask = returned_values
                        has_direct_tensors = True
                    else:
                        # Unexpected number of return values, fall back to regular sample
                        print(f"Warning: sample_tensors returned unexpected number of values: {len(returned_values)}")
                        transitions = self.memory.sample(self.batch_size)
                else:
                    # Not a tuple, fall back to regular sample
                    print(f"Warning: sample_tensors did not return a tuple: {type(returned_values)}")
                    transitions = self.memory.sample(self.batch_size)
            except Exception as e:
                # If we can't use sample_tensors, try the regular sample method
                print(f"Error using sample_tensors: {e}")
                print("Falling back to regular sample method")
                transitions = self.memory.sample(self.batch_size)
        else:
            # Use the regular sample method
            sample_result = self.memory.sample(self.batch_size)
            
            # Check if this is from a prioritized buffer (returns tuple of (transitions, indices, weights))
            if isinstance(sample_result, tuple) and len(sample_result) == 3:
                transitions, indices, importance_weights = sample_result
                has_prioritized = True
            else:
                transitions = sample_result
        
        # If we don't have direct tensors, we need to process the transitions
        if not has_direct_tensors and transitions is not None:
            # Create batch by combining transitions
            batch = Transition(*zip(*transitions))
            
            # Process states carefully - handle both tensor and numpy inputs
            states = []
            for s in batch.state:
                if s is not None:
                    # Make sure it's a tensor on the right device
                    if isinstance(s, torch.Tensor):
                        states.append(s.to(self.device))
                    elif isinstance(s, np.ndarray):
                        states.append(torch.tensor(s, dtype=torch.float32, device=self.device))
                    else:
                        # Handle unexpected types - convert to numpy first if possible
                        try:
                            s_array = np.array(s)
                            states.append(torch.tensor(s_array, dtype=torch.float32, device=self.device))
                        except:
                            print(f"Warning: Could not convert state of type {type(s)} to tensor")
                            # Skip this sample or use a dummy tensor
                            continue
            
            # Skip further processing if we don't have enough valid states
            if len(states) < 2:
                print(f"Warning: Not enough valid states ({len(states)}) to train. Skipping batch.")
                return None
                
            # Stack states into a batch tensor
            try:
                state_batch = torch.stack(states)
            except Exception as e:
                # If stacking fails, print more diagnostic info
                print(f"Error stacking states: {e}")
                print(f"Shapes: {[s.shape if hasattr(s, 'shape') else 'unknown' for s in states]}")
                return None
            
            # Handle actions - convert to tensor if needed
            if all(isinstance(a, torch.Tensor) for a in batch.action):
                action_batch = torch.cat([a.to(self.device) for a in batch.action])
            else:
                action_batch = torch.tensor([a if not isinstance(a, torch.Tensor) else a.item() 
                                            for a in batch.action], dtype=torch.long, device=self.device)
            
            # Handle rewards
            if all(isinstance(r, torch.Tensor) for r in batch.reward):
                reward_batch = torch.cat([r.to(self.device) for r in batch.reward])
            else:
                reward_batch = torch.tensor([r if not isinstance(r, torch.Tensor) else r.item() 
                                            for r in batch.reward], dtype=torch.float32, device=self.device)
            
            # Handle done flags
            if all(isinstance(d, torch.Tensor) for d in batch.done):
                done_batch = torch.cat([d.to(self.device) for d in batch.done])
                non_final_mask = ~done_batch
            else:
                non_final_mask = torch.tensor(
                    tuple(map(lambda d: not (d if not isinstance(d, torch.Tensor) else d.item()), batch.done)), 
                    device=self.device, dtype=torch.bool
                )
            
            # Get all non-final next states
            non_final_next_states = None
            if non_final_mask.any():
                non_final_states = []
                
                # We need to get next_states for non-terminal states
                valid_next_states = [(i, s) for i, (s, d) in enumerate(zip(batch.next_state, batch.done)) 
                                   if not (d if not isinstance(d, torch.Tensor) else d.item()) and s is not None]
                
                if valid_next_states:
                    for _, s in valid_next_states:
                        if isinstance(s, torch.Tensor):
                            non_final_states.append(s.to(self.device))
                        elif isinstance(s, np.ndarray):
                            non_final_states.append(torch.tensor(s, dtype=torch.float32, device=self.device))
                        else:
                            try:
                                s_array = np.array(s)
                                non_final_states.append(torch.tensor(s_array, dtype=torch.float32, device=self.device))
                            except:
                                # Skip this one
                                continue
                
                    if non_final_states:
                        try:
                            non_final_next_states = torch.stack(non_final_states)
                        except Exception as e:
                            print(f"Error stacking next_states: {e}")
                            print(f"Shapes: {[s.shape if hasattr(s, 'shape') else 'unknown' for s in non_final_states]}")
        
        # Make sure we have all the tensors we need
        if state_batch is None or action_batch is None or reward_batch is None or non_final_mask is None:
            print("Critical error: Missing required tensors after processing")
            return None
            
        # Mixed Precision training with autocast
        with torch.cuda.amp.autocast():
            # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
            state_action_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
            
            # Compute V(s_{t+1}) for all next states
            # Create tensor with correct dtype to match autocast precision
            next_state_values = torch.zeros(len(state_batch), device=self.device, dtype=torch.float32)

            if self.use_double_dqn and non_final_mask.any() and non_final_next_states is not None:
                # Double DQN: use policy_net to select action, target_net to evaluate it
                with torch.no_grad():
                    # Get actions from policy network
                    next_action_indices = self.policy_net(non_final_next_states).max(1)[1].unsqueeze(1)
                    # Evaluate Q-values using target network
                    target_q_values = self.target_net(non_final_next_states).gather(
                        1, next_action_indices).squeeze(1)
                    # Convert to Float32 to match destination
                    next_state_values[non_final_mask] = target_q_values.to(torch.float32)
            elif non_final_mask.any() and non_final_next_states is not None:
                # Standard DQN: use target_net for both selection and evaluation
                with torch.no_grad():
                    target_q_values = self.target_net(non_final_next_states).max(1)[0]
                    # Convert to Float32 to match destination
                    next_state_values[non_final_mask] = target_q_values.to(torch.float32)
            
            # Compute the expected Q values
            expected_state_action_values = reward_batch + (self.gamma * next_state_values)
            
            # Use Huber loss (smooth L1)
            criterion = nn.SmoothL1Loss(reduction='none')
            losses = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
            
            # Apply importance weights if using prioritized replay
            if importance_weights is not None:
                if isinstance(importance_weights, torch.Tensor):
                    losses = losses * importance_weights
                else:
                    try:
                        weights_tensor = torch.tensor(importance_weights, dtype=torch.float32, device=self.device)
                        losses = losses * weights_tensor.unsqueeze(1)
                    except Exception as e:
                        print(f"Error applying importance weights: {e}")
            
            loss = losses.mean()
            
            # Calculate TD errors for priority updates
            with torch.no_grad():
                td_errors = torch.abs(state_action_values - expected_state_action_values.unsqueeze(1))
        
        # Optimization step with mixed precision
        self.optimizer.zero_grad()
        
        # Use the GradScaler for mixed precision backpropagation
        self.scaler.scale(loss).backward()
        
        # Gradient clipping (after unscaling)
        if self.config.get("clip_gradients", True):
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.policy_net.parameters(), 
                self.config.get("max_grad_norm", 0.5)
            )
        
        # Update with scaler
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        # Update priorities in the replay buffer if using prioritized replay
        if has_prioritized and indices is not None and hasattr(self.memory, 'update_priorities'):
            # Get priorities from TD errors
            priorities = td_errors.detach().cpu().numpy()
            self.memory.update_priorities(indices, priorities)
        
        # Store loss for tracking
        loss_value = min(loss.item(), 50.0)  # Cap reported loss 
        self.loss_history.append(loss_value)
                
        return loss_value
        
    except Exception as e:
        print(f"Error in learn_amp: {e}")
        import traceback
        traceback.print_exc()
        return 50.0  # Return a default high loss value