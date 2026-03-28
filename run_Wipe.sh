#!/bin/bash

# Define a function to run training so we don't repeat code
TASK="TiltedWipe"
run_train() {
    ENV_NAME=$1
    ALGO=$2
    STEPS=$3
    USE_SPD=$4
    USE_LG=$5
    NUM_MARKERS=$6
    STIFF_PENALTY=$7
    EXP_NAME=$8
    SEED=$9
    KP_MIN=1
    KP_MAX=300

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
        --num_markers "$NUM_MARKERS"
        --stiff_penalty "$STIFF_PENALTY"
        --kp_min "$KP_MIN"
        --kp_max "$KP_MAX"
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
    python3 "${ARGS[@]}" > "logs/${ENV_NAME}_${ALGO}_${EXP_NAME}_SEED_${SEED}.log" 2>&1
    
    echo "Finished $ALGO on $ENV_NAME"
}

# Create logs directory if it doesn't exist
mkdir -p logs

STIFF_PENALTY=0.005
for NUM_MARKERS in 15 25  
do
    for SEED in 3 2 1 0 
    do
        run_train $TASK "SAC" 3_500_000 "TRUE" "TRUE" $NUM_MARKERS $STIFF_PENALTY "FULL_GRL_C0_NM${NUM_MARKERS}_SP${STIFF_PENALTY}_KP${KP_MIN}_${KP_MAX}" $SEED
        run_train $TASK "SAC" 3_500_000 "TRUE" "FALSE" $NUM_MARKERS $STIFF_PENALTY "SPD_ONLY_C0_NM${NUM_MARKERS}_SP${STIFF_PENALTY}_KP${KP_MIN}_${KP_MAX}" $SEED
        run_train $TASK "SAC" 3_500_000 "FALSE" "TRUE" $NUM_MARKERS $STIFF_PENALTY "LIE_ONLY_C0_NM${NUM_MARKERS}_SP${STIFF_PENALTY}_KP${KP_MIN}_${KP_MAX}" $SEED 
        run_train $TASK "SAC" 3_500_000 "FALSE" "FALSE" $NUM_MARKERS $STIFF_PENALTY "BASELINE_C0_NM${NUM_MARKERS}_SP${STIFF_PENALTY}_KP${KP_MIN}_${KP_MAX}" $SEED
    done
done

wait

echo "All Door experiments completed!"