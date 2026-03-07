#!/bin/bash

# Define a function to run training so we don't repeat code
TASK="NutAssemblySquare"
run_train() {
    ENV_NAME=$1
    ALGO=$2
    STEPS=$3
    SEED=$4
    USE_GRL="TRUE"
    USE_LG="TRUE"
    EXP_NAME="VIC_GRL_${USE_GRL}_LG_${USE_LG}_SEED_${SEED}"
    
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
        --seed $SEED \
        > logs/${ENV_NAME}_${ALGO}_${EXP_NAME}.log 2>&1
    
    echo "Finished $ALGO on $ENV_NAME"
}

# Create logs directory if it doesn't exist
mkdir -p logs

# --- 1. Door (Baseline) ---
for SEED in 1 2 3
do
    # run_train $TASK "PPO" 5_000_000
    # run_train $TASK "SAC" 5_000_000
    run_train $TASK "TQC" 5_000_000 $SEED

done

echo "All Door experiments completed!"