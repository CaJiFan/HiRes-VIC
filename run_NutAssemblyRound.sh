#!/bin/bash

# Define a function to run training so we don't repeat code
run_train() {
    ENV_NAME=$1
    ALGO=$2
    STEPS=$3
    EXP_NAME="GRL_OSC_VIC"
    
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

# run_train "NutAssemblyRound" "PPO" 3_000_000
# run_train "NutAssemblyRound" "SAC" 3_000_000
run_train "NutAssemblyRound" "TD3" 3_000_000
run_train "NutAssemblyRound" "TQC" 3_000_000

echo "All NutAssemblyRound experiments completed!"