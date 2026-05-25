#!/bin/bash

set -e

echo "🚀 Starting HiRes-VIC Hyperparameter Sweep 🚀"

# ✅ Create a logs directory if it doesn't exist
mkdir -p logs

# Fixed Parameters
ENVS=128
STEPS=1000000
SEEDS=(0 1 2 3)
GAMMAS=(0.95 0.90 0.99)

BASE_CMD="python src/train_pih.py --env PegInsertionSide-v1 --use_spd --record_video --n_envs $ENVS --total_timesteps $STEPS"

mkdir -p logs/pih

for seed in "${SEEDS[@]}"; do
    for gamma in "${GAMMAS[@]}"; do
        
        echo "=========================================================="
        echo "Running Seed: $seed | Gamma: $gamma"
        echo "=========================================================="

        # 2. LLM Prior (w = 0.2)
        RUN_NAME="spd_llm0.2_g${gamma}_fixedaxis_curriculum_s${seed}"
        echo "-> Starting: $RUN_NAME | Logging to logs/pih/${RUN_NAME}.log"
        $BASE_CMD --run_name $RUN_NAME --seed $seed --gamma $gamma \
            --use_llm_prior --llm_prior_weight 0.2 --llm_backend ollama > "logs/pih/${RUN_NAME}.log" 2>&1

        # 3. LLM Prior (w = 0.4)
        RUN_NAME="spd_llm0.4_g${gamma}_fixedaxis_curriculum_s${seed}"
        echo "-> Starting: $RUN_NAME | Logging to logs/pih/${RUN_NAME}.log"
        $BASE_CMD --run_name $RUN_NAME --seed $seed --gamma $gamma \
            --use_llm_prior --llm_prior_weight 0.4 --llm_backend ollama > "logs/pih/${RUN_NAME}.log" 2>&1

        # 1. Baseline
        RUN_NAME="spd_g${gamma}_fixedaxis_curriculum_s${seed}"
        echo "-> Starting: $RUN_NAME | Logging to logs/pih/${RUN_NAME}.log"
        $BASE_CMD --run_name $RUN_NAME --seed $seed --gamma $gamma > "logs/pih/${RUN_NAME}.log" 2>&1

    done
done

echo "🎉 Sweep Complete! Check WandB for the glorious curves."