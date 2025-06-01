# Progressive Training Script for Tetris DQN (PowerShell Version)
# This script implements the progressive testing approach from fix55.md
# 
# Usage: .\progressive_training.ps1 [-SkipTests] [-GpuOnly] [-ContinueFrom PHASE] [-PythonCmd COMMAND]
# 
# Phases:
# 1. Environment and component testing
# 2. Basic functionality verification (CPU, 10 episodes)
# 3. Short training run (CPU, 100 episodes) 
# 4. Medium training with GPU (1000 episodes)
# 5. Full training (10000+ episodes)

param(
    [switch]$SkipTests,
    [switch]$GpuOnly,
    [string]$ContinueFrom = "",
    [string]$PythonCmd = "python",
    [switch]$Help
)

# Show help if requested
if ($Help) {
    Write-Host "Usage: .\progressive_training.ps1 [-SkipTests] [-GpuOnly] [-ContinueFrom PHASE] [-PythonCmd COMMAND]"
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  -SkipTests           Skip initial test suite"
    Write-Host "  -GpuOnly            Only run GPU phases"
    Write-Host "  -ContinueFrom PHASE  Continue from specific phase (tests, basic, short, medium, full)"
    Write-Host "  -PythonCmd COMMAND   Python command to use (default: python)"
    Write-Host "  -Help               Show this help message"
    Write-Host ""
    Write-Host "Phases:"
    Write-Host "  tests   - Run test suite"
    Write-Host "  basic   - Basic functionality (10 episodes, CPU)"
    Write-Host "  short   - Short training (100 episodes, CPU)"
    Write-Host "  medium  - Medium training (1000 episodes, GPU)"
    Write-Host "  full    - Full training (10000+ episodes, GPU)"
    exit 0
}

# Logging functions
function Write-Info {
    param($Message)
    Write-Host "[INFO] $Message" -ForegroundColor Blue
}

function Write-Success {
    param($Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor Green
}

function Write-Warning {
    param($Message)
    Write-Host "[WARNING] $Message" -ForegroundColor Yellow
}

function Write-Error {
    param($Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

function Write-Phase {
    param($Message)
    Write-Host ""
    Write-Host "===================================================================================" -ForegroundColor Blue
    Write-Host "PHASE: $Message" -ForegroundColor Blue
    Write-Host "===================================================================================" -ForegroundColor Blue
    Write-Host ""
}

# Check if phase should be skipped
function Should-SkipPhase {
    param($Phase)
    
    if ($ContinueFrom) {
        switch ($ContinueFrom) {
            "tests" { return $false }
            "basic" { return $Phase -eq "tests" }
            "short" { return $Phase -in @("tests", "basic") }
            "medium" { return $Phase -in @("tests", "basic", "short") }
            "full" { return $Phase -ne "full" }
        }
    }
    
    if ($GpuOnly) {
        return $Phase -in @("tests", "basic", "short")
    }
    
    return $false
}

# Check Python and dependencies
function Test-Dependencies {
    Write-Info "Checking dependencies..."
    
    # Check Python
    try {
        $pythonVersion = & $PythonCmd --version 2>&1
        Write-Info "Using: $pythonVersion"
    }
    catch {
        Write-Error "Python not found. Please install Python or specify with -PythonCmd"
        exit 1
    }
    
    # Check PyTorch
    try {
        & $PythonCmd -c "import torch" 2>$null
        if ($LASTEXITCODE -ne 0) {
            throw "PyTorch import failed"
        }
    }
    catch {
        Write-Error "PyTorch not found. Please install PyTorch"
        exit 1
    }
    
    # Check CUDA availability
    try {
        $cudaAvailable = & $PythonCmd -c "import torch; print(torch.cuda.is_available())" 2>$null
        if ($cudaAvailable -eq "True") {
            $cudaDevice = & $PythonCmd -c "import torch; print(torch.cuda.get_device_name(0))" 2>$null
            Write-Success "CUDA available: $cudaDevice"
            $script:CudaAvailable = $true
        }
        else {
            Write-Warning "CUDA not available - will use CPU only"
            $script:CudaAvailable = $false
            if ($GpuOnly) {
                Write-Error "GPU-only mode requested but CUDA not available"
                exit 1
            }
        }
    }
    catch {
        Write-Warning "Could not check CUDA availability"
        $script:CudaAvailable = $false
    }
    
    Write-Success "Dependencies check passed"
}

# Run test suite
function Invoke-Tests {
    if (Should-SkipPhase "tests") {
        Write-Info "Skipping tests phase"
        return
    }
    
    Write-Phase "TESTING - Component Verification"
    
    if ($SkipTests) {
        Write-Warning "Skipping tests (-SkipTests flag used)"
        return
    }
    
    $testFiles = @(
        "test_environment.py",
        "test_preprocessing.py", 
        "test_agent.py",
        "test_memory.py",
        "test_integration.py"
    )
    
    foreach ($testFile in $testFiles) {
        Write-Info "Running $testFile..."
        & $PythonCmd $testFile
        if ($LASTEXITCODE -ne 0) {
            Write-Error "$testFile failed"
            exit 1
        }
    }
    
    Write-Success "All tests passed!"
}

# Create training script
function New-TrainingScript {
    param($ConfigType, $Episodes, $Device)
    
    $outputFile = "run_training_$ConfigType.py"
    
    $scriptContent = @"
#!/usr/bin/env python3
"""
Generated training script for $ConfigType configuration.
Episodes: $Episodes, Device: $Device
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
    print(f"TETRIS DQN TRAINING - $($ConfigType.ToUpper()) CONFIGURATION")
    print("=" * 60)
    print(f"Episodes: $Episodes")
    print(f"Device: $Device")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Load configuration
    if "$ConfigType" == "minimal":
        config = get_minimal_config()
    elif "$ConfigType" == "enhanced":
        config = get_enhanced_config()
    elif "$ConfigType" == "gpu":
        config = get_gpu_config()
    else:
        raise ValueError(f"Unknown config type: $ConfigType")
    
    # Override settings
    config["num_episodes"] = $Episodes
    config["device"] = "$Device"
    
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
    model_path = f"tetris_dqn_{ConfigType}_{Episodes}ep.pt"
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
        print("\n✅ Training completed successfully!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
"@
    
    $scriptContent | Out-File -FilePath $outputFile -Encoding UTF8
    return $outputFile
}

# Phase 1: Basic Functionality Test
function Invoke-BasicPhase {
    if (Should-SkipPhase "basic") {
        Write-Info "Skipping basic phase"
        return
    }
    
    Write-Phase "PHASE 1 - Basic Functionality (10 episodes, CPU)"
    
    Write-Info "Creating basic training script..."
    $script = New-TrainingScript "minimal" 10 "cpu"
    
    Write-Info "Running basic training test..."
    & $PythonCmd $script
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Basic training failed"
        exit 1
    }
    
    Write-Success "Basic functionality verified!"
}

# Phase 2: Short Training Run
function Invoke-ShortPhase {
    if (Should-SkipPhase "short") {
        Write-Info "Skipping short phase"
        return
    }
    
    Write-Phase "PHASE 2 - Short Training (100 episodes, CPU)"
    
    Write-Info "Creating short training script..."
    $script = New-TrainingScript "enhanced" 100 "cpu"
    
    Write-Info "Running short training..."
    & $PythonCmd $script
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Short training failed"
        exit 1
    }
    
    Write-Success "Short training completed successfully!"
}

# Phase 3: Medium Training with GPU
function Invoke-MediumPhase {
    if (Should-SkipPhase "medium") {
        Write-Info "Skipping medium phase"
        return
    }
    
    Write-Phase "PHASE 3 - Medium Training (1000 episodes, GPU)"
    
    if (-not $script:CudaAvailable) {
        Write-Warning "CUDA not available, running on CPU instead"
        $device = "cpu"
        $configType = "enhanced"
    }
    else {
        $device = "cuda"
        $configType = "gpu"
    }
    
    Write-Info "Creating medium training script..."
    $script = New-TrainingScript $configType 1000 $device
    
    Write-Info "Running medium training..."
    & $PythonCmd $script
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Medium training failed"
        exit 1
    }
    
    Write-Success "Medium training completed successfully!"
}

# Phase 4: Full Training
function Invoke-FullPhase {
    if (Should-SkipPhase "full") {
        Write-Info "Skipping full phase"
        return
    }
    
    Write-Phase "PHASE 4 - Full Training (10000+ episodes, GPU)"
    
    if (-not $script:CudaAvailable) {
        Write-Error "Full training requires CUDA. Please ensure GPU is available."
        exit 1
    }
    
    Write-Info "Creating full training script..."
    $script = New-TrainingScript "gpu" 10000 "cuda"
    
    Write-Info "Starting full training..."
    Write-Info "This may take several hours. Monitor GPU memory and temperature."
    Write-Info "You can stop training with Ctrl+C and resume later."
    
    & $PythonCmd $script
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Full training failed"
        exit 1
    }
    
    Write-Success "Full training completed successfully!"
}

# Main execution
function Main {
    Write-Host "===================================================================================" -ForegroundColor Blue
    Write-Host "TETRIS DQN PROGRESSIVE TRAINING SCRIPT" -ForegroundColor Blue
    Write-Host "===================================================================================" -ForegroundColor Blue
    Write-Host ""
    
    # Check dependencies
    Test-Dependencies
    
    # Run phases
    Invoke-Tests
    Invoke-BasicPhase
    Invoke-ShortPhase
    Invoke-MediumPhase
    Invoke-FullPhase
    
    Write-Host ""
    Write-Host "===================================================================================" -ForegroundColor Green
    Write-Host "ALL PHASES COMPLETED SUCCESSFULLY!" -ForegroundColor Green
    Write-Host "===================================================================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Your Tetris DQN training pipeline is now complete. You should have:"
    Write-Host "1. ✅ Verified all components work correctly"
    Write-Host "2. ✅ Basic training functionality"
    Write-Host "3. ✅ Short training run results"
    Write-Host "4. ✅ Medium training with GPU optimization"
    Write-Host "5. ✅ Full-scale training results"
    Write-Host ""
    Write-Host "Generated models:"
    Get-ChildItem -Filter "*.pt" -ErrorAction SilentlyContinue | ForEach-Object { Write-Host $_.Name }
    Write-Host ""
    Write-Host "Next steps:"
    Write-Host "- Analyze training results and metrics"
    Write-Host "- Adjust hyperparameters based on performance"
    Write-Host "- Experiment with different architectures"
    Write-Host "- Deploy the trained model for inference"
    Write-Host ""
    Write-Host "Happy training! 🎮🤖"
}

# Run main function
Main