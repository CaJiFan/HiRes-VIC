#!/bin/bash
#
# --- Slurm Configuration ---
#SBATCH --job-name=Nut_Square_VLM
#SBATCH --output=logs/nut_square_vlm_%j.txt
#SBATCH --error=logs/nut_square_vlm_%j.err
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
OLLAMA_PID=$!

cleanup() {
    if [ -n "$OLLAMA_PID" ]; then
        echo "Stopping background Ollama server (PID $OLLAMA_PID)..."
        kill "$OLLAMA_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

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
# 4. RL TRAINING FUNCTION (VLM-ENABLED)
# ==========================================
run_train() {
    ENV_NAME=$1
    USE_SPD=$2
    USE_LG=$3
    USE_VLM=$4
    STEPS=$5
    EXP_NAME=$6
    SEED=$7
    PRIOR_WEIGHT=$8
    VLM_MODEL=$9

    echo "=================================================="
    echo "Starting SAC on $ENV_NAME for $STEPS steps (VLM: $USE_VLM, Model: $VLM_MODEL)..."
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
    
    # Always enable LLM prior for VLM testing
    ARGS+=(--use_llm_prior)
    ARGS+=(--llm_backend ollama)
    ARGS+=(--llm_prior_weight "$PRIOR_WEIGHT")
    
    # Add VLM-specific arguments if enabled
    if [ "$USE_VLM" == "TRUE" ]; then
        ARGS+=(--use_vlm)
        ARGS+=(--vlm_model "$VLM_MODEL")
        # Reduced query interval for faster phase switching in VLM mode
        ARGS+=(--llm_query_interval 10)
        # Enable cameras explicitly
        ARGS+=(--use_cameras)
        # For NutAssemblySquare: robot0_eye_in_hand (end-effector) + frontview (scene)
        ARGS+=(--camera_names "robot0_eye_in_hand,frontview")
        ARGS+=(--vlm_image_size 224)
    else
        # Text-only: keep default query interval
        ARGS+=(--llm_query_interval 50)
        ARGS+=(--llm_model llama3.2)
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
    
    "${ARGS[@]}" > "logs/${ENV_NAME}_VLM_${EXP_NAME}_SEED_${SEED}.log" 2>&1 &
    echo "Launched: ${ENV_NAME} | ${EXP_NAME} | VLM=${USE_VLM} | Model=${VLM_MODEL} | SEED=${SEED}"
}

wait_for_queue() {
    while [ $(jobs -pr | wc -l) -gt 12 ]; do
        sleep 10
    done
}


# ==========================================
# 5. VLM EXPERIMENT EXECUTION
# ==========================================
# This script focuses on VLM performance across different models and configurations

TASK="NutAssemblySquare"
STEPS=10_000_000

echo "Starting VLM-guided residual RL experiments..."
echo "Testing: llama3.2 (text+multimodal), llava (vision specialist)"
echo ""

for SEED in 3 2 1; do
    for GAMMA in 0.99 0.90; do
        for LLM_WEIGHT in 0.25 0.50 0.75; do
            
            # ===== VLM Experiment 1: llama3.2 (native multimodal) =====
            wait_for_queue
            run_train $TASK "TRUE" "TRUE" "TRUE" $STEPS \
                "FULL_GRL_RES_10M_VLM_LLAMA32_${LLM_WEIGHT}_G${GAMMA}" \
                $SEED $LLM_WEIGHT "llama3.2"

            wait_for_queue
            run_train $TASK "FALSE" "FALSE" "TRUE" $STEPS \
                "BASELINE_RES_10M_VLM_LLAMA32_${LLM_WEIGHT}_G${GAMMA}" \
                $SEED $LLM_WEIGHT "llama3.2"

            # ===== VLM Experiment 2: LLaVA (vision specialist via ollama) =====
            # wait_for_queue
            # run_train $TASK "TRUE" "TRUE" "TRUE" $STEPS \
            #     "FULL_GRL_VLM_LLAVA_${LLM_WEIGHT}_G${GAMMA}" \
            #     $SEED $LLM_WEIGHT "llava"

            # wait_for_queue
            # run_train $TASK "FALSE" "FALSE" "TRUE" $STEPS \
            #     "BASELINE_RES_VLM_LLAVA_${LLM_WEIGHT}_G${GAMMA}" \
            #     $SEED $LLM_WEIGHT "llava"

            # ===== Baseline: Text-only LLM (for comparison) =====
            # wait_for_queue
            # run_train $TASK "TRUE" "TRUE" "FALSE" $STEPS \
            #     "FULL_GRL_TEXT_ONLY_${LLM_WEIGHT}_G${GAMMA}" \
            #     $SEED $LLM_WEIGHT "llama3.2"

            # wait_for_queue
            # run_train $TASK "FALSE" "FALSE" "FALSE" $STEPS \
            #     "BASELINE_RES_TEXT_ONLY_${LLM_WEIGHT}_G${GAMMA}" \
            #     $SEED $LLM_WEIGHT "llama3.2"

        done
    done
done

echo "All VLM experiments queued. Waiting for jobs to finish..."
wait
echo "✅ VLM experiments completed (jobs finished)."
echo ""
echo "Comparison structure:"
echo "  - FULL_GRL_RES_10M_VLM_LLAMA32: Full geometry (SPD+LG) + llama3.2 multimodal vision"
echo "  - BASELINE_RES_10M_VLM_LLAMA32: Baseline + llama3.2 multimodal vision"
echo "  - FULL_GRL_RES_10M_VLM_LLAVA: Full geometry (SPD+LG) + LLaVA specialist"
echo "  - BASELINE_RES_10M_VLM_LLAVA: Baseline + LLaVA specialist"
echo "  - FULL_GRL_RES_10M_TEXT_ONLY: Full geometry + text-only LLM (control)"
echo "  - BASELINE_RES_10M_TEXT_ONLY: Baseline + text-only LLM (control)"
