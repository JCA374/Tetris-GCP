#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<EOF
Usage: $0 [--no-nohup] [--no-shutdown]
  --no-nohup     Keep training in the foreground (live console output)
  --no-shutdown  Leave the VM running when done
EOF
  exit 1
}

# ─── 1) Parse flags ───────────────────────────────────────────────────────────────
USE_NOHUP=true
SHUTDOWN=true
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-nohup)    USE_NOHUP=false; shift ;;
    --no-shutdown) SHUTDOWN=false;  shift ;;
    -h|--help)     usage            ;;
    *)             break            ;;
  esac
done

# ─── 2) Threading & sizing ─────────────────────────────────────────────────────────
CORES=$(nproc)
export OMP_NUM_THREADS=$CORES
export MKL_NUM_THREADS=$CORES

PARALLEL_ENVS=$(( CORES * 2 ))    # ~2 envs per core
BATCH_SIZE=512                    # bigger GPU batches

# ─── 3) Other params & paths ──────────────────────────────────────────────────────
FULL_EP=25000
ANALYSIS_INTERVAL=3000
SCRIPT=run_gpu_training.py
CONFIG=config.py
FULL_LOG=full_training.log
ANALYSIS_OUT=training_analysis.txt

# ─── 4) Sanity + debug ─────────────────────────────────────────────────────────────
echo "[INFO] Debug:
  Cores          = $CORES
  Parallel Envs  = $PARALLEL_ENVS
  Batch Size     = $BATCH_SIZE
  Episodes       = $FULL_EP
  No‑nohup       = $([[ $USE_NOHUP == false ]] && echo yes || echo no)
  No‑shutdown    = $([[ $SHUTDOWN == false ]] && echo yes || echo no)"
[[ -f "$SCRIPT" ]] || { echo "ERROR: $SCRIPT not found"; exit 1; }
[[ -f "$CONFIG" ]] || { echo "ERROR: $CONFIG not found"; exit 1; }

# ─── 5) In‑place config tweaks ─────────────────────────────────────────────────────
echo "[INFO] Backing up & adjusting $CONFIG..."
cp "$CONFIG" "$CONFIG".backup

# learning rate, gamma, batch
sed -i 's|"learning_rate":[[:space:]]*[0-9.eE-]*\([,}]\)|"learning_rate": 1e-05\1|' "$CONFIG"
sed -i 's|"gamma":[[:space:]]*[0-9.eE-]*\([,}]\)|"gamma": 0.995\1|'         "$CONFIG"
sed -i 's|"batch_size":[[:space:]]*[0-9]*\([,}]\)|"batch_size": 512\1|'       "$CONFIG"

# exploration
sed -i 's|"epsilon_end":[[:space:]]*[0-9.eE-]*\([,}]\)|"epsilon_end": 0.01\1|'    "$CONFIG"
sed -i 's|"epsilon_decay":[[:space:]]*[0-9.eE-]*\([,}]\)|"epsilon_decay": 0.9999\1|' "$CONFIG"

# replay & prioritized‑replay
sed -i 's|"replay_capacity":[[:space:]]*[0-9]*\([,}]\)|"replay_capacity": 500000\1|' "$CONFIG"
sed -i 's|"pr_alpha":[[:space:]]*[0-9.eE-]*\([,}]\)|"pr_alpha": 0.6\1|'             "$CONFIG"
sed -i 's|"pr_beta_start":[[:space:]]*[0-9.eE-]*\([,}]\)|"pr_beta_start": 0.4\1|'   "$CONFIG"
sed -i 's|"pr_beta_frames":[[:space:]]*[0-9]*\([,}]\)|"pr_beta_frames": 200000\1|'  "$CONFIG"

# multi‑step returns
sed -i 's|"n_steps":[[:space:]]*[0-9]*\([,}]\)|"n_steps": 3\1|'                    "$CONFIG"

# reward shaping
sed -i 's|"reward_lines_cleared_weight":[[:space:]]*[0-9.eE-]*\([,}]\)|"reward_lines_cleared_weight": 1000.0\1|' "$CONFIG"
sed -i 's|"reward_tetris_bonus":[[:space:]]*[0-9.eE-]*\([,}]\)|"reward_tetris_bonus": 2000.0\1|'             "$CONFIG"
sed -i 's|"reward_survival":[[:space:]]*[0-9.eE-]*\([,}]\)|"reward_survival": 0.5\1|'                         "$CONFIG"
sed -i 's|"reward_scale":[[:space:]]*[0-9.eE-]*\([,}]\)|"reward_scale": 1.0\1|'                               "$CONFIG"

# target‑net updates
sed -i 's|"use_soft_update":[[:space:]]*False|"use_soft_update": True|' "$CONFIG"
sed -i 's|"tau":[[:space:]]*[0-9.eE-]*\([,}]\)|"tau": 0.005\1|'     "$CONFIG"
sed -i 's|"target_update":[[:space:]]*[0-9]*\([,}]\)|"target_update": 100\1|' "$CONFIG"

echo "[INFO] Config backup → $CONFIG.backup"

# ─── 6) Analysis & monitor funcs ─────────────────────────────────────────────────
run_analysis() {
  ep="$1"
  out="${ANALYSIS_OUT%.txt}_ep${ep}.txt"
  echo "[INFO] Running analysis @ ep $ep..."
  python analyze_dqn_log.py "$FULL_LOG" --plots --output "$out" \
    && echo "[INFO] Saved → $out" \
    || echo "[ERROR] Analysis failed @ ep $ep"
}

monitor_training() {
  last=0
  while :; do
    [[ -f "$FULL_LOG" ]] || { sleep 10; continue; }
    cur=$(grep -oE 'Episode ([0-9]+)/' "$FULL_LOG" \
           | tail -1 | grep -oE '[0-9]+' || echo 0)
    if (( cur >= last + ANALYSIS_INTERVAL )); then
      run_analysis "$cur"; last=$cur
    fi
    (( cur >= FULL_EP )) && break
    sleep 60
  done
}

# ─── 7) Kick off monitor ─────────────────────────────────────────────────────────
monitor_training & MON_PID=$!
echo "[INFO] Monitor PID = $MON_PID"

# ─── 8) Build Python command ─────────────────────────────────────────────────────
PYTHON=$(which python)
TRAIN_CMD="\
$PYTHON $SCRIPT \
  --episodes $FULL_EP \
  --parallel-envs $PARALLEL_ENVS \
  --batch-size $BATCH_SIZE \
  --log-file $FULL_LOG \
  --debug \
  --resume \
"

# ─── 9) Launch training ──────────────────────────────────────────────────────────
if [[ "$USE_NOHUP" == true ]]; then
  echo "[INFO] Launching training detached with nohup…"
  nohup bash -c "$TRAIN_CMD 2>&1 | tee -a training_progress.log" \
    > training_launch.log 2>&1 &
  TRAIN_PID=$!
  echo "[INFO] Training PID = $TRAIN_PID"
else
  echo "[INFO] Launching training in foreground…"
  bash -c "$TRAIN_CMD 2>&1 | tee -a training_progress.log"
fi

# ───10) Wait + final analysis + shutdown ────────────────────────────────────────
if [[ "$USE_NOHUP" == true ]]; then
  wait $TRAIN_PID
fi
wait $MON_PID
run_analysis "$FULL_EP"

echo "[INFO] Done. Flushing to disk…"
sync; sleep 5

if [[ "$SHUTDOWN" == true ]]; then
  echo "[INFO] Powering off VM now…"
  sudo shutdown -h now
else
  echo "[INFO] Shutdown skipped."
fi
