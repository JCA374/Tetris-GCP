"""
Minimal configuration for Tetris DQN with validation.

This configuration includes only essential parameters and provides
validation to ensure consistency and catch common configuration errors.
"""
import torch
import warnings
from typing import Dict, Any, List, Optional


def get_minimal_config() -> Dict[str, Any]:
    """
    Get the minimal configuration for Tetris DQN training.
    
    This contains only the essential parameters needed for basic training.
    Start with this configuration and gradually add features.
    
    Returns:
        Dictionary with minimal configuration parameters
    """
    return {
        # Core DQN settings
        "model_type": "dqn",  # Start with basic DQN
        "learning_rate": 1e-4,
        "batch_size": 32,
        "replay_capacity": 10000,
        "gamma": 0.99,
        
        # Exploration
        "epsilon_start": 1.0,
        "epsilon_end": 0.1,
        "epsilon_decay": 0.995,
        
        # Training
        "num_episodes": 100,
        "target_update": 10,  # Update target network every 10 episodes
        
        # Simple reward structure
        "reward_lines_cleared_weight": 1000.0,
        "reward_game_over_penalty": -100.0,
        
        # Disable advanced features initially
        "use_double_dqn": False,
        "use_prioritized_replay": False,
        "use_curriculum_learning": False,
        "use_enhanced_preprocessing": False,
        "use_amp": False,
        "use_lr_scheduler": False,
        
        # Basic device settings
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "num_envs": 1,  # Single environment for debugging
    }


def get_enhanced_config() -> Dict[str, Any]:
    """
    Get enhanced configuration with additional features enabled.
    
    Use this after validating that the minimal configuration works.
    
    Returns:
        Dictionary with enhanced configuration parameters
    """
    config = get_minimal_config()
    
    # Enable advanced features
    config.update({
        "model_type": "dueldqn",
        "use_double_dqn": True,
        "use_enhanced_preprocessing": True,
        "use_prioritized_replay": True,
        "batch_size": 128,
        "num_episodes": 1000,
        "num_envs": 4,
        
        # Learning rate scheduling
        "use_lr_scheduler": True,
        "lr_scheduler_step_size": 1000,
        "lr_scheduler_gamma": 0.5,
        
        # Learning rate warmup
        "lr_warmup_steps": 1000,
        "lr_warmup_lr": 1e-5,
        
        # Prioritized replay settings
        "per_alpha": 0.6,
        "per_beta": 0.4,
        
        # Gradient clipping
        "clip_gradients": True,
        "max_grad_norm": 5.0,
    })
    
    return config


def get_gpu_config() -> Dict[str, Any]:
    """
    Get configuration optimized for GPU training.
    
    Use this for full-scale training on GPU with all features enabled.
    
    Returns:
        Dictionary with GPU-optimized configuration parameters
    """
    config = get_enhanced_config()
    
    # GPU optimizations
    config.update({
        "batch_size": 512,
        "num_episodes": 10000,
        "num_envs": 32,
        "use_amp": True,  # Automatic Mixed Precision
        
        # Replay buffer
        "replay_capacity": 100000,
        
        # More sophisticated reward structure
        "use_curriculum_learning": True,
        
        # Soft target updates for stability
        "use_soft_update": True,
        "tau": 0.005,
    })
    
    return config


def validate_config(config: Dict[str, Any], fix_errors: bool = True) -> Dict[str, Any]:
    """
    Validate configuration parameters and check for common issues.
    
    Args:
        config: Configuration dictionary to validate
        fix_errors: Whether to automatically fix detected errors
        
    Returns:
        Validated (and possibly corrected) configuration dictionary
        
    Raises:
        ValueError: If critical errors are found and fix_errors=False
    """
    validated_config = config.copy()
    errors = []
    warnings_list = []
    
    # 1. Check required parameters
    required_params = [
        "model_type", "learning_rate", "batch_size", "gamma",
        "epsilon_start", "epsilon_end", "num_episodes"
    ]
    
    for param in required_params:
        if param not in validated_config:
            if fix_errors:
                # Set default values
                defaults = get_minimal_config()
                if param in defaults:
                    validated_config[param] = defaults[param]
                    warnings_list.append(f"Missing required parameter '{param}', set to default: {defaults[param]}")
                else:
                    errors.append(f"Missing required parameter '{param}' with no default available")
            else:
                errors.append(f"Missing required parameter '{param}'")
    
    # 2. Check parameter ranges and types
    validations = [
        ("learning_rate", float, (1e-6, 1e-1), "Learning rate should be between 1e-6 and 1e-1"),
        ("batch_size", int, (1, 2048), "Batch size should be between 1 and 2048"),
        ("gamma", float, (0.9, 0.999), "Gamma should be between 0.9 and 0.999"),
        ("epsilon_start", float, (0.1, 1.0), "Epsilon start should be between 0.1 and 1.0"),
        ("epsilon_end", float, (0.001, 0.5), "Epsilon end should be between 0.001 and 0.5"),
        ("num_episodes", int, (1, 100000), "Number of episodes should be between 1 and 100000"),
    ]
    
    for param, expected_type, (min_val, max_val), message in validations:
        if param in validated_config:
            value = validated_config[param]
            
            # Type check
            if not isinstance(value, expected_type):
                if fix_errors:
                    try:
                        validated_config[param] = expected_type(value)
                        warnings_list.append(f"Converted '{param}' from {type(value)} to {expected_type}")
                    except (ValueError, TypeError):
                        errors.append(f"Cannot convert '{param}' to {expected_type}: {value}")
                else:
                    errors.append(f"Parameter '{param}' should be {expected_type}, got {type(value)}")
                continue
            
            # Range check
            if not (min_val <= value <= max_val):
                if fix_errors:
                    validated_config[param] = max(min_val, min(max_val, value))
                    warnings_list.append(f"Clamped '{param}' to valid range [{min_val}, {max_val}]: {value} -> {validated_config[param]}")
                else:
                    errors.append(f"{message}: got {value}")
    
    # 3. Check logical consistency
    if "epsilon_start" in validated_config and "epsilon_end" in validated_config:
        if validated_config["epsilon_start"] <= validated_config["epsilon_end"]:
            if fix_errors:
                validated_config["epsilon_start"] = validated_config["epsilon_end"] + 0.1
                warnings_list.append("Fixed epsilon_start to be greater than epsilon_end")
            else:
                errors.append("epsilon_start should be greater than epsilon_end")
    
    # 4. Check conflicting settings
    conflicts = [
        (["use_prioritized_replay", "use_amp"], "Prioritized replay may not work well with AMP"),
        (["use_curriculum_learning", "num_envs"], "Curriculum learning works best with single environment"),
    ]
    
    for conflict_params, message in conflicts:
        if all(validated_config.get(param, False) for param in conflict_params):
            warnings_list.append(f"Potential conflict: {message}")
    
    # 5. Device-specific validations
    if validated_config.get("device") == "cuda" and not torch.cuda.is_available():
        if fix_errors:
            validated_config["device"] = "cpu"
            warnings_list.append("CUDA not available, switched to CPU")
        else:
            errors.append("CUDA device requested but not available")
    
    # 6. Check batch size compatibility with replay capacity
    if "batch_size" in validated_config and "replay_capacity" in validated_config:
        if validated_config["batch_size"] > validated_config["replay_capacity"]:
            if fix_errors:
                validated_config["replay_capacity"] = validated_config["batch_size"] * 10
                warnings_list.append(f"Increased replay_capacity to {validated_config['replay_capacity']} (10x batch_size)")
            else:
                errors.append("Batch size cannot be larger than replay capacity")
    
    # Report warnings
    for warning in warnings_list:
        warnings.warn(warning, UserWarning)
    
    # Handle errors
    if errors:
        error_message = "Configuration validation failed:\n" + "\n".join(f"- {error}" for error in errors)
        if fix_errors:
            warnings.warn(f"Some errors could not be auto-fixed:\n{error_message}", UserWarning)
        else:
            raise ValueError(error_message)
    
    return validated_config


def get_config_for_phase(phase: str) -> Dict[str, Any]:
    """
    Get configuration for different training phases.
    
    Args:
        phase: Training phase ('debug', 'basic', 'enhanced', 'production')
        
    Returns:
        Configuration dictionary for the specified phase
    """
    if phase == "debug":
        config = get_minimal_config()
        config.update({
            "num_episodes": 10,
            "replay_capacity": 1000,
            "batch_size": 16,
            "target_update": 5,
        })
    elif phase == "basic":
        config = get_minimal_config()
        config["num_episodes"] = 100
    elif phase == "enhanced":
        config = get_enhanced_config()
    elif phase == "production":
        config = get_gpu_config()
    else:
        raise ValueError(f"Unknown phase: {phase}. Choose from 'debug', 'basic', 'enhanced', 'production'")
    
    return validate_config(config)


def print_config_summary(config: Dict[str, Any]) -> None:
    """
    Print a human-readable summary of the configuration.
    
    Args:
        config: Configuration dictionary to summarize
    """
    print("=" * 50)
    print("CONFIGURATION SUMMARY")
    print("=" * 50)
    
    # Core settings
    print(f"Model Type: {config.get('model_type', 'unknown')}")
    print(f"Device: {config.get('device', 'unknown')}")
    print(f"Episodes: {config.get('num_episodes', 'unknown')}")
    print(f"Batch Size: {config.get('batch_size', 'unknown')}")
    print(f"Learning Rate: {config.get('learning_rate', 'unknown')}")
    
    # Advanced features
    features = []
    if config.get("use_double_dqn", False):
        features.append("Double DQN")
    if config.get("use_prioritized_replay", False):
        features.append("Prioritized Replay")
    if config.get("use_enhanced_preprocessing", False):
        features.append("Enhanced Preprocessing")
    if config.get("use_curriculum_learning", False):
        features.append("Curriculum Learning")
    if config.get("use_amp", False):
        features.append("Mixed Precision")
    if config.get("use_lr_scheduler", False):
        features.append("LR Scheduling")
    
    print(f"Features: {', '.join(features) if features else 'None'}")
    
    # Memory and performance
    print(f"Replay Capacity: {config.get('replay_capacity', 'unknown')}")
    print(f"Number of Envs: {config.get('num_envs', 'unknown')}")
    
    print("=" * 50)


# Pre-defined configurations for common use cases
CONFIGS = {
    "minimal": get_minimal_config(),
    "enhanced": get_enhanced_config(),
    "gpu": get_gpu_config(),
    "debug": get_config_for_phase("debug"),
}


if __name__ == "__main__":
    # Demo the configuration system
    print("Testing configuration validation...")
    
    # Test minimal config
    minimal = get_minimal_config()
    print("\nMinimal Configuration:")
    print_config_summary(minimal)
    
    # Test validation with errors
    bad_config = {
        "learning_rate": "not_a_number",  # Type error
        "batch_size": -5,  # Range error
        "epsilon_start": 0.05,  # Logic error (less than epsilon_end)
        "epsilon_end": 0.1,
    }
    
    print("\nTesting validation with auto-fix...")
    try:
        fixed_config = validate_config(bad_config, fix_errors=True)
        print("Fixed configuration:")
        print_config_summary(fixed_config)
    except Exception as e:
        print(f"Error: {e}")