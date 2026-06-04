#!/bin/bash
#
# --- Slurm Configuration ---
#SBATCH --job-name=Nut_Square
#SBATCH --output=logs/nut_square_%j.txt
#SBATCH --error=logs/nut_square_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48                    
#SBATCH --mem=150G                            
#SBATCH --time=48:00:00                       
#SBATCH --gres=gpu:lovelace:1                 

# ==========================================
# 1. ENVIRONMENT ACTIVATION
# ==========================================
# (Uncomment and adjust these to match your environment)
# module load anaconda3
# source $(conda info --base)/etc/profile.d/conda.sh
# conda activate your_rl_env

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


eval "$(conda shell.bash hook)"
conda activate tfm

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
    STEPS=$4
    EXP_NAME=$5
    SEED=$6
    PRIOR_WEIGHT=$7

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
        --use_llm_prior
        --llm_backend ollama
        --llm_prior_weight "$PRIOR_WEIGHT"
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
    
    # CRITICAL FIX: Added '&' so tasks actually run in parallel
    "${ARGS[@]}" > "logs/${ENV_NAME}_SAC_${EXP_NAME}_SEED_${SEED}.log" 2>&1 &
    echo "Launched: ${ENV_NAME} | ${EXP_NAME} | SEED=${SEED}"
}

wait_for_queue() {
    while [ $(jobs -p | wc -l) -ge 6 ]; do
        sleep 10
    done
}

TASK="NutAssemblySquare"
GAMMA=0.90
STEPS=3_000_000

for SEED in 3 2 1; do
    for LLM_WEIGHT in 0.2 0.4; do
        # run_train $TASK "TRUE" "TRUE" $STEPS "FULL_GRL_LLM${LLM_WEIGHT}_G${GAMMA}" $SEED
        # wait_for_queue

        # run_train $TASK "TRUE" "FALSE" $STEPS "SPD_ONLY_LLM${LLM_WEIGHT}_G${GAMMA}" $SEED
        # wait_for_queue

        run_train $TASK "FALSE" "TRUE" $STEPS "LIE_ONLY_LLM${LLM_WEIGHT}_G${GAMMA}" $SEED $LLM_WEIGHT
        wait_for_queue

        run_train $TASK "FALSE" "FALSE" $STEPS "BASELINE_LLM${LLM_WEIGHT}_G${GAMMA}" $SEED $LLM_WEIGHT
        wait_for_queue
    done

    
done

echo "All NutAssemblySquare experiments queued. Waiting for jobs to finish..."
wait
echo "✅ NutAssemblySquare experiments completed (jobs finished)."