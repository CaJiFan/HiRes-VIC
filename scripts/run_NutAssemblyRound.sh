#!/bin/bash

# Run NutAssemblyRound experiments using src/train_fixed.py
run_train() {
    ENV_NAME=$1
    USE_SPD=$2
    USE_LG=$3
    STEPS=$4
    EXP_NAME=$5
    SEED=$6

    echo "=================================================="
    echo "Starting SAC on $ENV_NAME for $STEPS steps..."
    echo "Date: $(date)"
    echo "=================================================="

    ARGS=(
        python3 src/train_fixed.py
        --env "$ENV_NAME"
        --run_name "$EXP_NAME"
        --total_timesteps "$STEPS"
        --seed "$SEED"
        --gamma "$GAMMA"
    )

    if [ "$USE_SPD" == "TRUE" ]; then
        ARGS+=(--use_spd)
    fi
    if [ "$USE_SPD" == "FIXED" ]; then
        ARGS+=(--use_fixed)
    fi
    if [ "$USE_LG" == "TRUE" ]; then
        ARGS+=(--use_lie)
    fi

    mkdir -p logs
    "${ARGS[@]}" > "logs/${ENV_NAME}_SAC_${EXP_NAME}_SEED_${SEED}.log" 2>&1 &
    echo "Launched: ${ENV_NAME} | ${EXP_NAME} | SEED=${SEED}"
}

wait_for_queue() {
    while [ $(jobs -p | wc -l) -ge 6 ]; do
        sleep 10
    done
}

TASK="NutAssemblyRound"
GAMMA=0.90
STEPS=5000000

for SEED in 1 2 3; do
    wait_for_queue
    run_train $TASK "TRUE" "TRUE" $STEPS "NUT_RD_FULL_GRL_KP1_300_GAMMA${GAMMA}_SEED${SEED}" $SEED

    wait_for_queue
    run_train $TASK "TRUE" "FALSE" $STEPS "NUT_RD_SPD_ONLY_GAMMA${GAMMA}_SEED${SEED}" $SEED

    wait_for_queue
    run_train $TASK "FALSE" "TRUE" $STEPS "NUT_RD_LIE_ONLY_GAMMA${GAMMA}_SEED${SEED}" $SEED

    wait_for_queue
    run_train $TASK "FALSE" "FALSE" $STEPS "NUT_RD_BASELINE_GAMMA${GAMMA}_SEED${SEED}" $SEED
done

echo "All NutAssemblyRound experiments queued. Waiting for jobs to finish..."
wait
echo "✅ NutAssemblyRound experiments completed (jobs finished)."