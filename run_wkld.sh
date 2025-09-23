#!/bin/bash

# Usage:
# ./run_workloads.sh           → use conf.ini
# or batch execution of triplets → provide workload, output, db_name

# Set project root directory (location of Utils.py)
PROJECT_ROOT="/path/to/your/project"
SCRIPT_DIR="$PROJECT_ROOT/PGUtils"

# Triplet combinations: each item is "<dataset_name> <workloads_name> <database_name>"
triplets=(
"job F_Uni_job_N10_W330.json imdb"
)


for triplet in "${triplets[@]}"; do
    echo ">>> Running: $triplet"

    read -r dataset_name workloads_name database_name <<< "$triplet"

    # Ensure all three parameters are provided
    if [[ -z "$dataset_name" || -z "$workloads_name" || -z "$database_name" ]]; then
        echo "❌ Incomplete parameters, must provide 3 values (workload, output, database)"
        exit 1
    fi

    python3 "$SCRIPT_DIR/loadMemCollectParallel.py" "$dataset_name" "$workloads_name" "$database_name"

    echo ">>> Waiting 20 seconds..."
    sleep 20
done

echo ">>> All triplets executed."
