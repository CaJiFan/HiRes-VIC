#!/bin/bash

# Define a function to run training so we don't repeat code
run_train() {
    ENV_NAME=$1
    ALGO=$2
    STEPS=$3
    EXP_NAME="VIC_FT"
    
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
# run_train "Door" "PPO" 500000
# run_train "Door" "SAC" 500000
run_train "Door" "TD3" 500000
run_train "Door" "TQC" 500000

# --- 2. NutAssemblySquare (Easy Alignment) ---
run_train "NutAssemblySquare" "PPO" 500000
run_train "NutAssemblySquare" "SAC" 500000
run_train "NutAssemblySquare" "TD3" 500000
run_train "NutAssemblySquare" "TQC" 500000

# --- 3. Wipe (Impedance / Force) ---
# Increased steps because force control is harder to learn
run_train "Wipe" "PPO" 1000000
run_train "Wipe" "SAC" 1000000
run_train "Wipe" "TD3" 1000000
run_train "Wipe" "TQC" 1000000

# --- 4. NutAssemblyRound (Precision Geometry) ---
# Hardest task, needs the most time
run_train "NutAssemblyRound" "PPO" 1000000
run_train "NutAssemblyRound" "SAC" 1000000
run_train "NutAssemblyRound" "TD3" 1000000
run_train "NutAssemblyRound" "TQC" 1000000

echo "All experiments completed!"