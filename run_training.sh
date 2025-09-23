#!/bin/bash

PROJECT_ROOT="/path/to/your/project/mpf"
PY_SCRIPT="$PROJECT_ROOT/src/training.py"
LOG_DIR="$PROJECT_ROOT/log"
mkdir -p "$LOG_DIR"

declare -A BATCH_JOBS_LIMITS=([8]=22 [16]=16 [32]=16 [64]=11)

MODELS=("")
JSON_PATHS=(
  "$PROJECT_ROOT/data/workload_file_1.json"
  "$PROJECT_ROOT/data/workload_file_2.json"
  "$PROJECT_ROOT/data/workload_file_3.json"
  "$PROJECT_ROOT/data/workload_file_4.json"
)

for BATCH_SIZE in 32; do
  MAX_PARALLEL=${BATCH_JOBS_LIMITS[$BATCH_SIZE]}
  echo ">>> Running batch size $BATCH_SIZE (max $MAX_PARALLEL concurrent jobs)"

  JOB_COUNT=0

  for JSON in "${JSON_PATHS[@]}"; do
    NAME=$(basename "$JSON" .json)

    for MODEL in "${MODELS[@]}"; do
      LOG="$LOG_DIR/${MODEL}_${NAME}_bz${BATCH_SIZE}TIME.log"
      nohup python -u "$PY_SCRIPT" "$MODEL" "$JSON" "$BATCH_SIZE" > "$LOG" 2>&1 &

      ((JOB_COUNT++))
      if (( JOB_COUNT % MAX_PARALLEL == 0 )); then
        wait
      fi
    done
  done

  wait
  echo ">>> Finished batch size $BATCH_SIZE"
done