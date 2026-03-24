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
        --num_markers 25
        --use_condensed_obj_obs  # Enable condensed object observation representation for all experiments
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

for SEED in 0 1 2 3
do
    run_train $TASK "SAC" 3_500_000 "TRUE" "TRUE" "FULL_GRL_CONDENSED_25_MARKERS" $SEED 
    run_train $TASK "SAC" 3_500_000 "TRUE" "FALSE" "SPD_ONLY_CONDENSED_25_MARKERS" $SEED 
    run_train $TASK "SAC" 3_500_000 "FALSE" "TRUE" "LIE_ONLY_CONDENSED_25_MARKERS" $SEED 
    run_train $TASK "SAC" 3_500_000 "FALSE" "FALSE" "BASELINE_CONDENSED_25_MARKERS" $SEED 
done


wait

echo "All Door experiments completed!"