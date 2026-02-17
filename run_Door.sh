#!/bin/bash

# Define a function to run training so we don't repeat code
run_train() {
    ENV_NAME=$1
    ALGO=$2
    STEPS=$3
    EXP_NAME="VIC_5M"
    
    echo "=================================================="
    echo "Starting $ALGO on $ENV_NAME for $STEPS steps..."
    echo "Date: $(date)"
    echo "=================================================="

    # Run Python script
    # > logs/... saves the text output to a file
    # 2>&1 captures errors too
    python3 scripts/train.py \
        --env $ENV_NAME \
        --algorithm $ALGO \
        --total_timesteps $STEPS \
        --run_name $EXP_NAME \
        > logs/${ENV_NAME}_${ALGO}_${EXP_NAME}.log 2>&1
    
    echo "Finished $ALGO on $ENV_NAME"
}

# Create logs directory if it doesn't exist
mkdir -p logs

# --- 1. Door (Baseline) ---
run_train "Door" "PPO" 500000
run_train "Door" "SAC" 500000
run_train "Door" "TD3" 500000
run_train "Door" "TQC" 500000

echo "All Door experiments completed!"