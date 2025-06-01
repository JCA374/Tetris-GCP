#!/bin/bash

# Progressive Training Script for Tetris DQN
# This script implements the progressive testing approach from fix55.md
# 
# Usage: ./progressive_training.sh [--skip-tests] [--gpu-only] [--continue-from=PHASE]
# 
# Phases:
# 1. Environment and component testing
# 2. Basic functionality verification (CPU, 10 episodes)
# 3. Short training run (CPU, 100 episodes) 
# 4. Medium training with GPU (1000 episodes)
# 5. Full training (10000+ episodes)

set -e  # Exit on any error

# Default settings
SKIP_TESTS=false
GPU_ONLY=false
CONTINUE_FROM=""
PYTHON_CMD="python"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        --gpu-only)
            GPU_ONLY=true
            shift
            ;;
        --continue-from=*)
            CONTINUE_FROM="${1#*=}"
            shift
            ;;
        --python=*)
            PYTHON_CMD="${1#*=}"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--skip-tests] [--gpu-only] [--continue-from=PHASE] [--python=COMMAND]"
            echo ""
            echo "Options:"
            echo "  --skip-tests           Skip initial test suite"
            echo "  --gpu-only            Only run GPU phases"
            echo "  --continue-from=PHASE  Continue from specific phase (tests, basic, short, medium, full)"
            echo "  --python=COMMAND      Python command to use (default: python)"
            echo "  -h, --help            Show this help message"
            echo ""
            echo "Phases:"
            echo "  tests   - Run test suite"
            echo "  basic   - Basic functionality (10 episodes, CPU)"
            echo "  short   - Short training (100 episodes, CPU)"
            echo "  medium  - Medium training (1000 episodes, GPU)"
            echo "  full    - Full training (10000+ episodes, GPU)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_phase() {
    echo ""
    echo "==================================================================================="
    echo -e "${BLUE}PHASE: $1${NC}"
    echo "==================================================================================="
    echo ""
}

# Check if phase should be skipped
should_skip_phase() {
    local phase=$1
    if [[ -n "$CONTINUE_FROM" ]]; then
        case "$CONTINUE_FROM" in
            "tests")
                return 1  # Don't skip any phase
                ;;
            "basic")
                [[ "$phase" == "tests" ]] && return 0 || return 1
                ;;
            "short")
                [[ "$phase" == "tests" || "$phase" == "basic" ]] && return 0 || return 1
                ;;
            "medium")
                [[ "$phase" == "tests" || "$phase" == "basic" || "$phase" == "short" ]] && return 0 || return 1
                ;;
            "full")
                [[ "$phase" != "full" ]] && return 0 || return 1
                ;;
        esac
    fi
    
    if [[ "$GPU_ONLY" == "true" ]]; then
        [[ "$phase" == "tests" || "$phase" == "basic" || "$phase" == "short" ]] && return 0 || return 1
    fi
    
    return 1
}

# Check Python and dependencies
check_dependencies() {
    log_info "Checking dependencies..."
    
    # Check Python
    if ! command -v $PYTHON_CMD &> /dev/null; then
        log_error "Python not found. Please install Python or specify with --python="
        exit 1
    fi
    
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1)
    log_info "Using: $PYTHON_VERSION"
    
    # Check PyTorch
    if ! $PYTHON_CMD -c "import torch" &> /dev/null; then
        log_error "PyTorch not found. Please install PyTorch"
        exit 1
    fi
    
    # Check CUDA availability
    CUDA_AVAILABLE=$($PYTHON_CMD -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")
    if [[ "$CUDA_AVAILABLE" == "True" ]]; then
        CUDA_DEVICE=$($PYTHON_CMD -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "Unknown")
        log_success "CUDA available: $CUDA_DEVICE"
    else
        log_warning "CUDA not available - will use CPU only"
        if [[ "$GPU_ONLY" == "true" ]]; then
            log_error "GPU-only mode requested but CUDA not available"
            exit 1
        fi
    fi
    
    log_success "Dependencies check passed"
}

# Run test suite
run_tests() {
    if should_skip_phase "tests"; then
        log_info "Skipping tests phase"
        return 0
    fi
    
    log_phase "TESTING - Component Verification"
    
    if [[ "$SKIP_TESTS" == "true" ]]; then
        log_warning "Skipping tests (--skip-tests flag used)"
        return 0
    fi
    
    log_info "Running environment tests..."
    if ! $PYTHON_CMD test_environment.py; then
        log_error "Environment tests failed"
        exit 1
    fi
    
    log_info "Running preprocessing tests..."
    if ! $PYTHON_CMD test_preprocessing.py; then
        log_error "Preprocessing tests failed"
        exit 1
    fi
    
    log_info "Running agent tests..."
    if ! $PYTHON_CMD test_agent.py; then
        log_error "Agent tests failed"
        exit 1
    fi
    
    log_info "Running memory tests..."
    if ! $PYTHON_CMD test_memory.py; then
        log_error "Memory tests failed"
        exit 1
    fi
    
    log_info "Running integration tests..."
    if ! $PYTHON_CMD test_integration.py; then
        log_error "Integration tests failed"
        exit 1
    fi
    
    log_success "All tests passed!"
}

# Create training script
create_training_script() {
    local config_type=$1
    local episodes=$2
    local device=$3
    local output_file="run_training_${config_type}.py"
    
    cat > "$output_file" << EOF
#!/usr/bin/env python3
"""
Generated training script for $config_type configuration.
Episodes: $episodes, Device: $device
"""
import sys
import os
import time
import torch
import numpy as np

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simple_tetris_env import SimpleTetrisEnv
from agent import DQNAgent
from preprocessing import preprocess_state
from minimal_config import get_minimal_config, get_enhanced_config, get_gpu_config, validate_config

def main():
    print("=" * 60)
    print(f"TETRIS DQN TRAINING - {config_type.upper()} CONFIGURATION")
    print("=" * 60)
    print(f"Episodes: $episodes")
    print(f"Device: $device")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Load configuration
    if "$config_type" == "minimal":
        config = get_minimal_config()
    elif "$config_type" == "enhanced":
        config = get_enhanced_config()
    elif "$config_type" == "gpu":
        config = get_gpu_config()
    else:
        raise ValueError(f"Unknown config type: $config_type")
    
    # Override settings
    config["num_episodes"] = $episodes
    config["device"] = "$device"
    
    # Validate configuration
    config = validate_config(config)
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Create environment
    env = SimpleTetrisEnv()
    
    # Determine input shape based on preprocessing
    if config.get("use_enhanced_preprocessing", False):
        input_shape = (4, 14, 7)
    else:
        input_shape = (1, 14, 7)
    
    # Create agent
    agent = DQNAgent(
        input_shape=input_shape,
        n_actions=env.action_space.n,
        device=config["device"],
        config=config
    )
    
    print(f"Agent created with {sum(p.numel() for p in agent.policy_net.parameters())} parameters")
    print()
    
    # Training loop
    episode_rewards = []
    episode_lengths = []
    lines_cleared_total = []
    
    start_time = time.time()
    
    for episode in range(config["num_episodes"]):
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        episode_lines_cleared = 0
        
        while True:
            # Preprocess state
            processed_state = preprocess_state(
                state, 
                include_piece_info=config.get("use_enhanced_preprocessing", False),
                device=config["device"]
            )
            
            # Select action
            action = agent.select_action(processed_state, training=True)
            
            # Environment step
            next_state, reward, done, info = env.step(action)
            
            # Store transition
            if next_state is not None:
                next_processed_state = preprocess_state(
                    next_state,
                    include_piece_info=config.get("use_enhanced_preprocessing", False),
                    device=config["device"]
                )
            else:
                next_processed_state = None
            
            agent.memory.push(processed_state, action, next_processed_state, reward, done)
            
            # Learn
            if len(agent.memory) >= agent.batch_size:
                loss = agent.learn()
            
            # Update metrics
            episode_reward += reward
            episode_length += 1
            
            if 'lines_cleared' in info:
                episode_lines_cleared += info['lines_cleared']
            
            if done:
                break
            
            state = next_state
        
        # Record episode metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        lines_cleared_total.append(episode_lines_cleared)
        
        # Update exploration
        agent.epsilon = max(agent.epsilon_end, agent.epsilon * agent.epsilon_decay)
        
        # Logging
        if episode % max(1, config["num_episodes"] // 10) == 0 or episode == config["num_episodes"] - 1:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_length = np.mean(episode_lengths[-100:])
            avg_lines = np.mean(lines_cleared_total[-100:])
            elapsed = time.time() - start_time
            
            print(f"Episode {episode:4d} | "
                  f"Reward: {episode_reward:6.1f} | "
                  f"Avg(100): {avg_reward:6.1f} | "
                  f"Length: {episode_length:3d} | "
                  f"Lines: {episode_lines_cleared:2d} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Time: {elapsed:.0f}s")
    
    # Final statistics
    print()
    print("=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)
    print(f"Total episodes: {len(episode_rewards)}")
    print(f"Average reward: {np.mean(episode_rewards):.2f}")
    print(f"Best episode reward: {np.max(episode_rewards):.2f}")
    print(f"Total lines cleared: {np.sum(lines_cleared_total)}")
    print(f"Average lines per episode: {np.mean(lines_cleared_total):.2f}")
    print(f"Training time: {(time.time() - start_time) / 60:.1f} minutes")
    
    # Save model
    model_path = f"tetris_dqn_{config_type}_{episodes}ep.pt"
    agent.save(model_path)
    print(f"Model saved to: {model_path}")
    
    return {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'lines_cleared': lines_cleared_total,
        'config': config
    }

if __name__ == "__main__":
    try:
        result = main()
        print("\\n✅ Training completed successfully!")
        sys.exit(0)
    except Exception as e:
        print(f"\\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
EOF
    
    chmod +x "$output_file"
    echo "$output_file"
}

# Phase 1: Basic Functionality Test
run_basic_phase() {
    if should_skip_phase "basic"; then
        log_info "Skipping basic phase"
        return 0
    fi
    
    log_phase "PHASE 1 - Basic Functionality (10 episodes, CPU)"
    
    log_info "Creating basic training script..."
    script=$(create_training_script "minimal" 10 "cpu")
    
    log_info "Running basic training test..."
    if ! $PYTHON_CMD "$script"; then
        log_error "Basic training failed"
        exit 1
    fi
    
    log_success "Basic functionality verified!"
}

# Phase 2: Short Training Run
run_short_phase() {
    if should_skip_phase "short"; then
        log_info "Skipping short phase"
        return 0
    fi
    
    log_phase "PHASE 2 - Short Training (100 episodes, CPU)"
    
    log_info "Creating short training script..."
    script=$(create_training_script "enhanced" 100 "cpu")
    
    log_info "Running short training..."
    if ! $PYTHON_CMD "$script"; then
        log_error "Short training failed"
        exit 1
    fi
    
    log_success "Short training completed successfully!"
}

# Phase 3: Medium Training with GPU
run_medium_phase() {
    if should_skip_phase "medium"; then
        log_info "Skipping medium phase"
        return 0
    fi
    
    log_phase "PHASE 3 - Medium Training (1000 episodes, GPU)"
    
    if [[ "$CUDA_AVAILABLE" != "True" ]]; then
        log_warning "CUDA not available, running on CPU instead"
        device="cpu"
        config_type="enhanced"
    else
        device="cuda"
        config_type="gpu"
    fi
    
    log_info "Creating medium training script..."
    script=$(create_training_script "$config_type" 1000 "$device")
    
    log_info "Running medium training..."
    if ! $PYTHON_CMD "$script"; then
        log_error "Medium training failed"
        exit 1
    fi
    
    log_success "Medium training completed successfully!"
}

# Phase 4: Full Training
run_full_phase() {
    if should_skip_phase "full"; then
        log_info "Skipping full phase"
        return 0
    fi
    
    log_phase "PHASE 4 - Full Training (10000+ episodes, GPU)"
    
    if [[ "$CUDA_AVAILABLE" != "True" ]]; then
        log_error "Full training requires CUDA. Please ensure GPU is available."
        exit 1
    fi
    
    log_info "Creating full training script..."
    script=$(create_training_script "gpu" 10000 "cuda")
    
    log_info "Starting full training..."
    log_info "This may take several hours. Monitor GPU memory and temperature."
    log_info "You can stop training with Ctrl+C and resume later."
    
    if ! $PYTHON_CMD "$script"; then
        log_error "Full training failed"
        exit 1
    fi
    
    log_success "Full training completed successfully!"
}

# Main execution
main() {
    echo "==================================================================================="
    echo "TETRIS DQN PROGRESSIVE TRAINING SCRIPT"
    echo "==================================================================================="
    echo ""
    
    # Check dependencies
    check_dependencies
    
    # Run phases
    run_tests
    run_basic_phase
    run_short_phase
    run_medium_phase
    run_full_phase
    
    echo ""
    echo "==================================================================================="
    echo -e "${GREEN}ALL PHASES COMPLETED SUCCESSFULLY!${NC}"
    echo "==================================================================================="
    echo ""
    echo "Your Tetris DQN training pipeline is now complete. You should have:"
    echo "1. ✅ Verified all components work correctly"
    echo "2. ✅ Basic training functionality"
    echo "3. ✅ Short training run results"
    echo "4. ✅ Medium training with GPU optimization"
    echo "5. ✅ Full-scale training results"
    echo ""
    echo "Generated models:"
    ls -la *.pt 2>/dev/null || echo "No model files found"
    echo ""
    echo "Next steps:"
    echo "- Analyze training results and metrics"
    echo "- Adjust hyperparameters based on performance"
    echo "- Experiment with different architectures"
    echo "- Deploy the trained model for inference"
    echo ""
    echo "Happy training! 🎮🤖"
}

# Run main function
main "$@"