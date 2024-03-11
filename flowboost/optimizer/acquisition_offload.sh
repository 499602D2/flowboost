#!/bin/bash

# Arguments (all required)
OPTIMIZER="$1"
MODEL_SNAPSHOT="$2"
DATA_SNAPSHOT="$3"
NUM_TRIALS="$4"
OUTPUT_PATH="$5"

python3 -m flowboost.optimizer.acquisition_offload \
    --optimizer "$OPTIMIZER" \
    --model_snapshot "$MODEL_SNAPSHOT" \
    --data_snapshot "$DATA_SNAPSHOT" \
    --num_trials "$NUM_TRIALS" \
    --output_path "$OUTPUT_PATH"
