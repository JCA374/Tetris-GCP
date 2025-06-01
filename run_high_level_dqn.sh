#!/bin/bash
# Script to run the high-level action Tetris DQN implementation

# Default values
EPISODES=5000
PARALLEL_ENVS=8
BATCH_SIZE=256
GRID_WIDTH=7
GRID_HEIGHT=14
MODE="train"
MODEL_PATH="models/high_level_final_model.pt"
EVAL_EPISODES=20
RENDER=false
VISUALIZE=false

# Help function
function show_help {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -m, --mode <train|test|compare>  Run mode (default: train)"
    echo "  -e, --episodes <num>             Number of training episodes (default: 5000)"
    echo "  -p, --parallel <num>             Number of parallel environments (default: 8)"
    echo "  -b, --batch-size <num>           Batch size (default: 256)"
    echo "  --width <num>                    Grid width (default: 7)"
    echo "  --height <num>                   Grid height (default: 14)"
    echo "  --model <path>                   Model path for evaluation (default: models/high_level_final_model.pt)"
    echo "  --eval-episodes <num>            Evaluation episodes (default: 20)"
    echo "  -r, --render                     Enable rendering (for evaluation mode)"
    echo "  -v, --visualize                  Visualize high-level actions (for evaluation mode)"
    echo "  -h, --help                       Show this help message"
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--mode)
            MODE="$2"
            shift 2
            ;;
        -e|--episodes)
            EPISODES="$2"
            shift 2
            ;;
        -p|--parallel)
            PARALLEL_ENVS="$2"
            shift 2
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --width)
            GRID_WIDTH="$2"
            shift 2
            ;;
        --height)
            GRID_HEIGHT="$2"
            shift 2
            ;;
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --eval-episodes)
            EVAL_EPISODES="$2"
            shift 2
            ;;
        -r|--render)
            RENDER=true
            shift
            ;;
        -v|--visualize)
            VISUALIZE=true
            shift
            ;;
        -h|--help)
            show_help
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            ;;
    esac
done

# Create necessary directories
mkdir -p models
mkdir -p checkpoints
mkdir -p logs

# Check for Python
if ! command -v python &> /dev/null; then
    echo "Error: Python is not installed or not in PATH"
    exit 1
fi

# Display run configuration
echo "===== High-Level Action Tetris DQN ====="
echo "Mode: $MODE"
echo "Grid dimensions: ${GRID_WIDTH}x${GRID_HEIGHT}"

case "$MODE" in
    train)
        echo "Training with:"
        echo "- Episodes: $EPISODES"
        echo "- Parallel environments: $PARALLEL_ENVS"
        echo "- Batch size: $BATCH_SIZE"
        
        # Run training with high-level actions
        python run_gpu_training_updated.py \
            --episodes "$EPISODES" \
            --parallel-envs "$PARALLEL_ENVS" \
            --batch-size "$BATCH_SIZE" \
            --grid-width "$GRID_WIDTH" \
            --grid-height "$GRID_HEIGHT" \
            --high-level-actions \
            --log-file "logs/high_level_training.log"
        ;;
        
    test)
        echo "Evaluating model: $MODEL_PATH"
        echo "- Evaluation episodes: $EVAL_EPISODES"
        
        RENDER_FLAG=""
        if [ "$RENDER" = true ]; then
            RENDER_FLAG="--render"
            echo "- Rendering enabled"
        fi
        
        VISUALIZE_FLAG=""
        if [ "$VISUALIZE" = true ]; then
            VISUALIZE_FLAG="--visualize-high-level"
            echo "- Visualization enabled"
        fi
        
        # Run evaluation
        python evaluate_models.py \
            --high-level-model "$MODEL_PATH" \
            --episodes "$EVAL_EPISODES" \
            --width "$GRID_WIDTH" \
            --height "$GRID_HEIGHT" \
            $RENDER_FLAG $VISUALIZE_FLAG
        ;;
        
    compare)
        echo "Running comparison between low-level and high-level actions"
        echo "- Episodes: $EPISODES"
        echo "- Parallel environments: $PARALLEL_ENVS"
        echo "- Batch size: $BATCH_SIZE"
        
        # Run comparison
        python training_comparison.py \
            --episodes "$EPISODES" \
            --parallel-envs "$PARALLEL_ENVS" \
            --batch-size "$BATCH_SIZE"
        ;;
        
    *)
        echo "Error: Unknown mode '$MODE'"
        show_help
        ;;
esac

echo "Done!"