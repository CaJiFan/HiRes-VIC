#!/bin/bash

# Define a function to run training so we don't repeat code
TASK="TiltedWipe"
run_train() {
    ENV_NAME=$1
    ALGO=$2
    STEPS=$3
    USE_SPD=$4
    USE_LG=$5
    EXP_NAME=$6
    SEED=$7
    
    echo "=================================================="
    echo "Starting $ALGO on $ENV_NAME for $STEPS steps..."
    echo "Date: $(date)"
    echo "=================================================="

    # 1. Start with the base arguments
    ARGS=(
        scripts/train.py
        --env "$ENV_NAME"
        --algorithm "$ALGO"
        --total_timesteps "$STEPS"
        --run_name "$EXP_NAME"
        --seed "$SEED"
    )

    # 2. Add conditional flags
    if [ "$USE_SPD" = "TRUE" ]; then
        ARGS+=(--use_spd)
    fi

    if [ "$USE_LG" = "TRUE" ]; then
        ARGS+=(--use_lie)
    fi

    # 3. Execute the command
    python3 "${ARGS[@]}" > "logs/${ENV_NAME}_${ALGO}_${EXP_NAME}_SPD_${USE_SPD}_LG_${USE_LG}_SEED_${SEED}.log" 2>&1
    
    echo "Finished $ALGO on $ENV_NAME"
}

# Create logs directory if it doesn't exist
mkdir -p logs

# --- 1. Door (Baseline) ---
# for SEED in 1 2 3
# do
#     # run_train $TASK "PPO" 1_000_000
#     # run_train $TASK "SAC" 1_000_000
#     # run_train $TASK "TD3" 1_000_000
#     run_train $TASK "TQC" 5_000_000 "FALSE" "FALSE" "VIC_TILTED" $SEED
# done

SEED=3
TASK="TiltedWipe"
run_train $TASK "SAC" 5_000_000 "TRUE" "TRUE" "VIC_TILTED" $SEED
run_train $TASK "SAC" 5_000_000 "TRUE" "FALSE" "VIC_TILTED" $SEED
run_train $TASK "SAC" 5_000_000 "FALSE" "TRUE" "VIC_TILTED" $SEED
run_train $TASK "SAC" 5_000_000 "FALSE" "FALSE" "VIC_TILTED" $SEED


echo "All Door experiments completed!"