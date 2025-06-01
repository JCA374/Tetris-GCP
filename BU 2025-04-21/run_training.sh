#!/usr/bin/env bash
set -euo pipefail

FULL_LOG="$1"; SCRIPT="$2"; ANALYSIS_OUTPUT="$3"; ANALYSIS_INTERVAL="$4"; FULL_EP="$5"; SHUTDOWN_FLAG="$6"
shift 6
SCRIPT_ARGS=("$@")

run_analysis() {
  local ep=$1
  local out="\${ANALYSIS_OUTPUT%.txt}_ep\${ep}.txt"
  echo "[INFO] Analyzing ep \$ep..."
  python analyze_dqn_log.py "\$FULL_LOG" --plots --output "\$out" >/dev/null 2>&1 \
    && echo "[INFO] Saved \$out" \
    || echo "[ERROR] Analysis failed for ep \$ep"
}

monitor_training() {
  local last=0
  while true; do
    [[ -f "\$FULL_LOG" ]] || { sleep 10; continue; }
    local cur=\$(grep -oE 'Episode ([0-9]+)/' "\$FULL_LOG" | tail -1 | grep -oE '[0-9]+' || echo 0)
    if (( cur >= last + ANALYSIS_INTERVAL )); then
      run_analysis "\$cur"; last=\$cur
    fi
    (( cur >= FULL_EP )) && break
    sleep 60
  done
}

monitor_training & MON_PID=\$!
echo "[INFO] Monitor PID: \$MON_PID"

if python "\$SCRIPT" "\${SCRIPT_ARGS[@]}" 2>&1 | tee -a training_progress.log; then
  echo "[INFO] Training finished"
else
  echo "[ERROR] Training crashed"
fi

wait \$MON_PID
run_analysis "\$FULL_EP"

echo "[INFO] All done, flushing..."
sync; sleep 30

if [[ "\$SHUTDOWN_FLAG" == "true" ]]; then
  echo "[INFO] Shutting down..."
  sudo shutdown -h now
else
  echo "[INFO] Shutdown skipped."
fi
