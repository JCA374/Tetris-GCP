@echo off
REM Progressive Training Script for Tetris DQN (Windows Batch Version)
REM This script implements the progressive testing approach from fix55.md
REM 
REM Usage: progressive_training.bat [skip-tests] [gpu-only] [continue-from=PHASE] [python=COMMAND]
REM 
REM Phases:
REM 1. Environment and component testing
REM 2. Basic functionality verification (CPU, 10 episodes)
REM 3. Short training run (CPU, 100 episodes) 
REM 4. Medium training with GPU (1000 episodes)
REM 5. Full training (10000+ episodes)

setlocal enabledelayedexpansion

REM Default settings
set "SKIP_TESTS=false"
set "GPU_ONLY=false"
set "CONTINUE_FROM="
set "PYTHON_CMD=python"
set "CUDA_AVAILABLE=false"

REM Parse command line arguments
:parse_args
if "%~1"=="" goto args_done
if "%~1"=="skip-tests" (
    set "SKIP_TESTS=true"
    shift
    goto parse_args
)
if "%~1"=="gpu-only" (
    set "GPU_ONLY=true"
    shift
    goto parse_args
)
if "%~1"=="help" goto show_help
if "%~1"=="-h" goto show_help
if "%~1"=="--help" goto show_help

REM Handle continue-from=value
set "arg=%~1"
if "!arg:~0,13!"=="continue-from=" (
    set "CONTINUE_FROM=!arg:~13!"
    shift
    goto parse_args
)

REM Handle python=value
if "!arg:~0,7!"=="python=" (
    set "PYTHON_CMD=!arg:~7!"
    shift
    goto parse_args
)

echo Unknown option: %~1
echo Use 'help' for usage information
exit /b 1

:show_help
echo Usage: progressive_training.bat [skip-tests] [gpu-only] [continue-from=PHASE] [python=COMMAND]
echo.
echo Options:
echo   skip-tests           Skip initial test suite
echo   gpu-only            Only run GPU phases
echo   continue-from=PHASE  Continue from specific phase (tests, basic, short, medium, full)
echo   python=COMMAND       Python command to use (default: python)
echo   help, -h, --help     Show this help message
echo.
echo Phases:
echo   tests   - Run test suite
echo   basic   - Basic functionality (10 episodes, CPU)
echo   short   - Short training (100 episodes, CPU)
echo   medium  - Medium training (1000 episodes, GPU)
echo   full    - Full training (10000+ episodes, GPU)
exit /b 0

:args_done

REM Logging functions (using labels)
goto main

:log_info
echo [INFO] %~1
goto :eof

:log_success
echo [SUCCESS] %~1
goto :eof

:log_warning
echo [WARNING] %~1
goto :eof

:log_error
echo [ERROR] %~1
goto :eof

:log_phase
echo.
echo ===================================================================================
echo PHASE: %~1
echo ===================================================================================
echo.
goto :eof

REM Check if phase should be skipped
:should_skip_phase
set "phase=%~1"
set "skip_result=false"

if "%CONTINUE_FROM%"=="basic" if "%phase%"=="tests" set "skip_result=true"
if "%CONTINUE_FROM%"=="short" if "%phase%"=="tests" set "skip_result=true"
if "%CONTINUE_FROM%"=="short" if "%phase%"=="basic" set "skip_result=true"
if "%CONTINUE_FROM%"=="medium" if "%phase%"=="tests" set "skip_result=true"
if "%CONTINUE_FROM%"=="medium" if "%phase%"=="basic" set "skip_result=true"
if "%CONTINUE_FROM%"=="medium" if "%phase%"=="short" set "skip_result=true"
if "%CONTINUE_FROM%"=="full" if not "%phase%"=="full" set "skip_result=true"

if "%GPU_ONLY%"=="true" (
    if "%phase%"=="tests" set "skip_result=true"
    if "%phase%"=="basic" set "skip_result=true"
    if "%phase%"=="short" set "skip_result=true"
)

goto :eof

REM Check Python and dependencies
:check_dependencies
call :log_info "Checking dependencies..."

REM Check Python
%PYTHON_CMD% --version >nul 2>&1
if errorlevel 1 (
    call :log_error "Python not found. Please install Python or specify with python=COMMAND"
    exit /b 1
)

for /f "tokens=*" %%i in ('%PYTHON_CMD% --version 2^>^&1') do set "PYTHON_VERSION=%%i"
call :log_info "Using: !PYTHON_VERSION!"

REM Check PyTorch
%PYTHON_CMD% -c "import torch" >nul 2>&1
if errorlevel 1 (
    call :log_error "PyTorch not found. Please install PyTorch"
    exit /b 1
)

REM Check CUDA availability
for /f "tokens=*" %%i in ('%PYTHON_CMD% -c "import torch; print(torch.cuda.is_available())" 2^>nul') do set "CUDA_CHECK=%%i"
if "%CUDA_CHECK%"=="True" (
    for /f "tokens=*" %%i in ('%PYTHON_CMD% -c "import torch; print(torch.cuda.get_device_name(0))" 2^>nul') do set "CUDA_DEVICE=%%i"
    call :log_success "CUDA available: !CUDA_DEVICE!"
    set "CUDA_AVAILABLE=true"
) else (
    call :log_warning "CUDA not available - will use CPU only"
    set "CUDA_AVAILABLE=false"
    if "%GPU_ONLY%"=="true" (
        call :log_error "GPU-only mode requested but CUDA not available"
        exit /b 1
    )
)

call :log_success "Dependencies check passed"
goto :eof

REM Create training script
:create_training_script
set "config_type=%~1"
set "episodes=%~2"
set "device=%~3"
set "output_file=run_training_%config_type%.py"

REM Create Python training script
(
echo #!/usr/bin/env python3
echo """
echo Generated training script for %config_type% configuration.
echo Episodes: %episodes%, Device: %device%
echo """
echo import sys
echo import os
echo import time
echo import torch
echo import numpy as np
echo.
echo # Add current directory to path
echo sys.path.insert^(0, os.path.dirname^(os.path.abspath^(__file__^)^)^)
echo.
echo from simple_tetris_env import SimpleTetrisEnv
echo from agent import DQNAgent
echo from preprocessing import preprocess_state
echo from minimal_config import get_minimal_config, get_enhanced_config, get_gpu_config, validate_config
echo.
echo def main^(^):
echo     print^("=" * 60^)
echo     print^(f"TETRIS DQN TRAINING - %config_type:~0,1%%config_type:~1%.upper^(^)% CONFIGURATION"^)
echo     print^("=" * 60^)
echo     print^(f"Episodes: %episodes%"^)
echo     print^(f"Device: %device%"^)
echo     print^(f"Time: {time.strftime^('%%Y-%%m-%%d %%H:%%M:%%S'^)}"^)
echo     print^(^)
echo     
echo     # Load configuration
echo     if "%config_type%" == "minimal":
echo         config = get_minimal_config^(^)
echo     elif "%config_type%" == "enhanced":
echo         config = get_enhanced_config^(^)
echo     elif "%config_type%" == "gpu":
echo         config = get_gpu_config^(^)
echo     else:
echo         raise ValueError^(f"Unknown config type: %config_type%"^)
echo     
echo     # Override settings
echo     config["num_episodes"] = %episodes%
echo     config["device"] = "%device%"
echo     
echo     # Validate configuration
echo     config = validate_config^(config^)
echo     
echo     print^("Configuration:"^)
echo     for key, value in config.items^(^):
echo         print^(f"  {key}: {value}"^)
echo     print^(^)
echo     
echo     # Create environment
echo     env = SimpleTetrisEnv^(^)
echo     
echo     # Determine input shape based on preprocessing
echo     if config.get^("use_enhanced_preprocessing", False^):
echo         input_shape = ^(4, 14, 7^)
echo     else:
echo         input_shape = ^(1, 14, 7^)
echo     
echo     # Create agent
echo     agent = DQNAgent^(
echo         input_shape=input_shape,
echo         n_actions=env.action_space.n,
echo         device=config["device"],
echo         config=config
echo     ^)
echo     
echo     print^(f"Agent created with {sum^(p.numel^(^) for p in agent.policy_net.parameters^(^)^)} parameters"^)
echo     print^(^)
echo     
echo     # Training loop
echo     episode_rewards = []
echo     episode_lengths = []
echo     lines_cleared_total = []
echo     
echo     start_time = time.time^(^)
echo     
echo     for episode in range^(config["num_episodes"]^):
echo         state = env.reset^(^)
echo         episode_reward = 0
echo         episode_length = 0
echo         episode_lines_cleared = 0
echo         
echo         while True:
echo             # Preprocess state
echo             processed_state = preprocess_state^(
echo                 state, 
echo                 include_piece_info=config.get^("use_enhanced_preprocessing", False^),
echo                 device=config["device"]
echo             ^)
echo             
echo             # Select action
echo             action = agent.select_action^(processed_state, training=True^)
echo             
echo             # Environment step
echo             next_state, reward, done, info = env.step^(action^)
echo             
echo             # Store transition
echo             if next_state is not None:
echo                 next_processed_state = preprocess_state^(
echo                     next_state,
echo                     include_piece_info=config.get^("use_enhanced_preprocessing", False^),
echo                     device=config["device"]
echo                 ^)
echo             else:
echo                 next_processed_state = None
echo             
echo             agent.memory.push^(processed_state, action, next_processed_state, reward, done^)
echo             
echo             # Learn
echo             if len^(agent.memory^) ^>= agent.batch_size:
echo                 loss = agent.learn^(^)
echo             
echo             # Update metrics
echo             episode_reward += reward
echo             episode_length += 1
echo             
echo             if 'lines_cleared' in info:
echo                 episode_lines_cleared += info['lines_cleared']
echo             
echo             if done:
echo                 break
echo             
echo             state = next_state
echo         
echo         # Record episode metrics
echo         episode_rewards.append^(episode_reward^)
echo         episode_lengths.append^(episode_length^)
echo         lines_cleared_total.append^(episode_lines_cleared^)
echo         
echo         # Update exploration
echo         agent.epsilon = max^(agent.epsilon_end, agent.epsilon * agent.epsilon_decay^)
echo         
echo         # Logging
echo         if episode %% max^(1, config["num_episodes"] // 10^) == 0 or episode == config["num_episodes"] - 1:
echo             avg_reward = np.mean^(episode_rewards[-100:]^)
echo             avg_length = np.mean^(episode_lengths[-100:]^)
echo             avg_lines = np.mean^(lines_cleared_total[-100:]^)
echo             elapsed = time.time^(^) - start_time
echo             
echo             print^(f"Episode {episode:4d} | "
echo                   f"Reward: {episode_reward:6.1f} | "
echo                   f"Avg^(100^): {avg_reward:6.1f} | "
echo                   f"Length: {episode_length:3d} | "
echo                   f"Lines: {episode_lines_cleared:2d} | "
echo                   f"Epsilon: {agent.epsilon:.3f} | "
echo                   f"Time: {elapsed:.0f}s"^)
echo     
echo     # Final statistics
echo     print^(^)
echo     print^("=" * 60^)
echo     print^("TRAINING COMPLETED"^)
echo     print^("=" * 60^)
echo     print^(f"Total episodes: {len^(episode_rewards^)}"^)
echo     print^(f"Average reward: {np.mean^(episode_rewards^):.2f}"^)
echo     print^(f"Best episode reward: {np.max^(episode_rewards^):.2f}"^)
echo     print^(f"Total lines cleared: {np.sum^(lines_cleared_total^)}"^)
echo     print^(f"Average lines per episode: {np.mean^(lines_cleared_total^):.2f}"^)
echo     print^(f"Training time: {^(time.time^(^) - start_time^) / 60:.1f} minutes"^)
echo     
echo     # Save model
echo     model_path = f"tetris_dqn_%config_type%_%episodes%ep.pt"
echo     agent.save^(model_path^)
echo     print^(f"Model saved to: {model_path}"^)
echo     
echo     return {
echo         'episode_rewards': episode_rewards,
echo         'episode_lengths': episode_lengths,
echo         'lines_cleared': lines_cleared_total,
echo         'config': config
echo     }
echo.
echo if __name__ == "__main__":
echo     try:
echo         result = main^(^)
echo         print^("\n✅ Training completed successfully!"^)
echo         sys.exit^(0^)
echo     except Exception as e:
echo         print^(f"\n❌ Training failed: {e}"^)
echo         import traceback
echo         traceback.print_exc^(^)
echo         sys.exit^(1^)
) > "%output_file%"

echo %output_file%
goto :eof

REM Run test suite
:run_tests
call :should_skip_phase "tests"
if "%skip_result%"=="true" (
    call :log_info "Skipping tests phase"
    goto :eof
)

call :log_phase "TESTING - Component Verification"

if "%SKIP_TESTS%"=="true" (
    call :log_warning "Skipping tests (skip-tests flag used)"
    goto :eof
)

for %%f in (test_environment.py test_preprocessing.py test_agent.py test_memory.py test_integration.py) do (
    call :log_info "Running %%f..."
    %PYTHON_CMD% %%f
    if errorlevel 1 (
        call :log_error "%%f failed"
        exit /b 1
    )
)

call :log_success "All tests passed!"
goto :eof

REM Phase 1: Basic Functionality Test
:run_basic_phase
call :should_skip_phase "basic"
if "%skip_result%"=="true" (
    call :log_info "Skipping basic phase"
    goto :eof
)

call :log_phase "PHASE 1 - Basic Functionality (10 episodes, CPU)"

call :log_info "Creating basic training script..."
call :create_training_script "minimal" "10" "cpu"
set "script=%output_file%"

call :log_info "Running basic training test..."
%PYTHON_CMD% "%script%"
if errorlevel 1 (
    call :log_error "Basic training failed"
    exit /b 1
)

call :log_success "Basic functionality verified!"
goto :eof

REM Phase 2: Short Training Run
:run_short_phase
call :should_skip_phase "short"
if "%skip_result%"=="true" (
    call :log_info "Skipping short phase"
    goto :eof
)

call :log_phase "PHASE 2 - Short Training (100 episodes, CPU)"

call :log_info "Creating short training script..."
call :create_training_script "enhanced" "100" "cpu"
set "script=%output_file%"

call :log_info "Running short training..."
%PYTHON_CMD% "%script%"
if errorlevel 1 (
    call :log_error "Short training failed"
    exit /b 1
)

call :log_success "Short training completed successfully!"
goto :eof

REM Phase 3: Medium Training with GPU
:run_medium_phase
call :should_skip_phase "medium"
if "%skip_result%"=="true" (
    call :log_info "Skipping medium phase"
    goto :eof
)

call :log_phase "PHASE 3 - Medium Training (1000 episodes, GPU)"

if "%CUDA_AVAILABLE%"=="false" (
    call :log_warning "CUDA not available, running on CPU instead"
    set "device=cpu"
    set "config_type=enhanced"
) else (
    set "device=cuda"
    set "config_type=gpu"
)

call :log_info "Creating medium training script..."
call :create_training_script "!config_type!" "1000" "!device!"
set "script=%output_file%"

call :log_info "Running medium training..."
%PYTHON_CMD% "%script%"
if errorlevel 1 (
    call :log_error "Medium training failed"
    exit /b 1
)

call :log_success "Medium training completed successfully!"
goto :eof

REM Phase 4: Full Training
:run_full_phase
call :should_skip_phase "full"
if "%skip_result%"=="true" (
    call :log_info "Skipping full phase"
    goto :eof
)

call :log_phase "PHASE 4 - Full Training (10000+ episodes, GPU)"

if "%CUDA_AVAILABLE%"=="false" (
    call :log_error "Full training requires CUDA. Please ensure GPU is available."
    exit /b 1
)

call :log_info "Creating full training script..."
call :create_training_script "gpu" "10000" "cuda"
set "script=%output_file%"

call :log_info "Starting full training..."
call :log_info "This may take several hours. Monitor GPU memory and temperature."
call :log_info "You can stop training with Ctrl+C and resume later."

%PYTHON_CMD% "%script%"
if errorlevel 1 (
    call :log_error "Full training failed"
    exit /b 1
)

call :log_success "Full training completed successfully!"
goto :eof

REM Main execution
:main
echo ===================================================================================
echo TETRIS DQN PROGRESSIVE TRAINING SCRIPT
echo ===================================================================================
echo.

REM Check dependencies
call :check_dependencies

REM Run phases
call :run_tests
call :run_basic_phase
call :run_short_phase
call :run_medium_phase
call :run_full_phase

echo.
echo ===================================================================================
echo ALL PHASES COMPLETED SUCCESSFULLY!
echo ===================================================================================
echo.
echo Your Tetris DQN training pipeline is now complete. You should have:
echo 1. ✅ Verified all components work correctly
echo 2. ✅ Basic training functionality
echo 3. ✅ Short training run results
echo 4. ✅ Medium training with GPU optimization
echo 5. ✅ Full-scale training results
echo.
echo Generated models:
dir /b *.pt 2>nul
echo.
echo Next steps:
echo - Analyze training results and metrics
echo - Adjust hyperparameters based on performance
echo - Experiment with different architectures
echo - Deploy the trained model for inference
echo.
echo Happy training! 🎮🤖