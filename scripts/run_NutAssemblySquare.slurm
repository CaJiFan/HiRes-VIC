#!/bin/bash
#
# --- Slurm Configuration ---
#SBATCH --job-name=Nut_Square
#SBATCH --output=logs/nut_square_%j.txt
#SBATCH --error=logs/nut_square_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=100                    
#SBATCH --mem=160G                            
#SBATCH --time=168:00:00
#SBATCH --gres=gpu:lovelace:1                 

# ==========================================
# 1. ENVIRONMENT ACTIVATION
# ==========================================
eval "$(conda shell.bash hook)"
conda activate tfm

# ==========================================
# 2. OLLAMA SERVER SETUP
# ==========================================
# Tell Ollama to save massive weights to your generous /home drive
export OLLAMA_MODELS="/home/cjimenez/ollama_models"

# Generate a unique port to avoid conflicts with other jobs on the same node
UNIQUE_PORT=$((11000 + (SLURM_JOB_ID % 1000)))
export OLLAMA_HOST="127.0.0.1:${UNIQUE_PORT}"

echo "Starting Ollama server on port ${UNIQUE_PORT}..."
~/.local/bin/ollama serve &

# Wait for the server to fully initialize
sleep 10

# ==========================================
# 3. SMOKE TEST
# ==========================================
echo "=================================================="
echo "SMOKE TEST: Verifying Ollama is responding..."
echo "=================================================="

RESPONSE=$(curl -s -X POST http://${OLLAMA_HOST}/api/generate -d '{
  "model": "llama3.2:latest",
  "prompt": "Say hello!",
  "stream": false
}')

# Check if the JSON contains "done":true, which proves the API works perfectly
if echo "$RESPONSE" | grep -q '"done":true'; then
    echo "✅ LLM Smoke Test PASSED! The model is loaded and generating text."
    echo "Model Output:"
    echo "$RESPONSE" | grep -o '"response":"[^"]*"'
else
    echo "❌ LLM Smoke Test FAILED! Ollama is not responding correctly."
    echo "Raw Output: $RESPONSE"
    echo "Aborting job to save compute hours."
    exit 1
fi
echo "=================================================="




# --- Admin Required L40S Fixes ---
export NCCL_P2P_DISABLE=1
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024
export MALLOC_ARENA_MAX=1

# ==========================================
# 4. RL TRAINING FUNCTION
# ==========================================
run_train() {
    ENV_NAME=$1
    USE_SPD=$2
    USE_LG=$3
    USE_LLM_PRIOR=$4
    STEPS=$5
    EXP_NAME=$6
    SEED=$7
    PRIOR_WEIGHT=$8

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
        --record_video
    )
    if [ "$USE_LLM_PRIOR" == "TRUE" ]; then
        ARGS+=(--use_llm_prior)
        ARGS+=(--llm_backend ollama)
        ARGS+=(--llm_prior_weight "$PRIOR_WEIGHT")
    fi
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
    while [ $(jobs -p | wc -l) -gt 12 ]; do
        sleep 10
    done
}

# ==========================================
# 5. EXPERIMENT EXECUTION
# ==========================================
TASK="NutAssemblySquare"
# GAMMA=0.90
STEPS=10_000_000

for SEED in 3 2 1; do
    for GAMMA in 0.99 0.90; do
        for LLM_WEIGHT in 0.25 0.50 0.75; do
            wait_for_queue
            run_train $TASK "TRUE" "TRUE" "TRUE" $STEPS "FULL_GRL_RES_10M_LLM${LLM_WEIGHT}_G${GAMMA}" $SEED $LLM_WEIGHT

            # wait_for_queue
            # run_train $TASK "TRUE" "FALSE" "TRUE" $STEPS "SPD_ONLY_LLM${LLM_WEIGHT}_G${GAMMA}" $SEED $LLM_WEIGHT

            # wait_for_queue
            # run_train $TASK "FALSE" "TRUE" "TRUE" $STEPS "LIE_ONLY_KP1_300_G${GAMMA}" $SEED $LLM_WEIGHT

            wait_for_queue
            run_train $TASK "FALSE" "FALSE" "TRUE" $STEPS "BASELINE_RES_10M_LLM${LLM_WEIGHT}_G${GAMMA}" $SEED $LLM_WEIGHT

        done

        # wait_for_queue
        # run_train $TASK "TRUE" "TRUE" "FALSE" $STEPS "FULL_GRL_10M_G${GAMMA}" $SEED 0

        # wait_for_queue
        # run_train $TASK "TRUE" "FALSE" "FALSE" $STEPS "SPD_ONLY_G${GAMMA}" $SEED 0

        # wait_for_queue
        # run_train $TASK "FALSE" "TRUE" "FALSE" $STEPS "LIE_ONLY_KP1_300_G${GAMMA}" $SEED 0

        # wait_for_queue
        # run_train $TASK "FALSE" "FALSE" "FALSE" $STEPS "BASELINE_10M_KP1_300_G${GAMMA}" $SEED 0
    done
done

echo "All NutAssemblySquare experiments queued. Waiting for jobs to finish..."
wait
echo "✅ NutAssemblySquare experiments completed (jobs finished)."