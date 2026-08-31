#!/bin/bash
#
# --- Slurm Configuration ---
#SBATCH --job-name=wipev0.7
#SBATCH --output=logs/wipev0.7_%j.txt
#SBATCH --error=logs/wipev0.7_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:lovelace:1


set -e

eval "$(conda shell.bash hook)"
conda activate tfm

# Force MuJoCo EGL rendering to use the SLURM-assigned GPU to prevent VRAM splitting
export MUJOCO_GL="egl"
export MUJOCO_EGL_DEVICE_ID=$(python3 -c "
import os, glob
try:
    cuda_id = os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0]
    pci_full = os.popen(f'nvidia-smi --id={cuda_id} --query-gpu=pci.bus_id --format=csv,noheader').read().strip().lower()
    pci_short = pci_full.split(':')[-2] + ':' + pci_full.split(':')[-1]
    
    for i, node in enumerate(sorted(glob.glob('/sys/class/drm/renderD*/device'))):
        drm_pci = os.path.basename(os.path.realpath(node)).lower()
        if pci_short in drm_pci:
            print(i)
            exit(0)
except Exception:
    pass
print(0)
")

mkdir -p logs/wipe

# --- Admin Required L40S Fixes ---
export NCCL_P2P_DISABLE=1
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024
export MALLOC_ARENA_MAX=1

# ── Fixed hypers ──────────────────────────────────────────────────────────────
TASK="TiltedWipe"
STEPS=1000000
HORIZON=150
N_ENVS=16
GAMMA=0.95
# GAMMA=$(awk "BEGIN {print 1 - 1/$HORIZON}")
NUM_MARKERS=5
SEEDS=(0 1 2 3 4)
LR=3e-4            # lowered from 3e-4 to prevent stability drops on the impedance manifold
BATCH_SIZE=1024    # increased from 512 for smoother gradient steps
GAMMA_START=0.95   # curriculum: start here, anneal to GAMMA by 50% of training


# ── LLM-related hparams ────────────────────────────────────────────────────────
LLM_W_INIT=0.70
# LLM_W_FLOOR=0.50
LLM_W_FLOOR=$LLM_W_INIT         
LLM_ANNEAL_SCHEDULE="cosine"
# LLM_ANNEAL=$((STEPS * 50 / 100 / N_ENVS))
LLM_ANNEAL=0
LLM_BACKEND="ollama"
LLM_MODEL="llama3.2"
# LLM_MODEL="qwen2.5:3b"
LLM_QUERY_INTERVAL=10

wait_for_queue() {
    while [ $(jobs -pr | wc -l) -ge 6 ]; do
        sleep 10
    done
}

# ── OLLAMA SERVER SETUP ───────────────────────────────────────────────────────
export OLLAMA_MODELS="/home/cjimenez/ollama_models"
UNIQUE_PORT=$((11000 + (SLURM_JOB_ID % 1000)))
export OLLAMA_HOST="127.0.0.1:${UNIQUE_PORT}"
export OLLAMA_NUM_PARALLEL=5
export OLLAMA_MAX_LOADED_MODELS=1

OLLAMA_RESPONSE=""
check_ollama() {
    OLLAMA_RESPONSE=$(curl -s -X POST "http://${OLLAMA_HOST}/api/generate" -d '{
        "model": "'"${LLM_MODEL}"'",
        "prompt": "Reply with one word: ready",
        "stream": false
    }' 2>/dev/null)
    echo "$OLLAMA_RESPONSE" | grep -q '"done":true' 
}

# ── Helper functions ──────────────────────────────────────────────────────────
run_llm() {
    # Geometric + LLM semantic prior run
    local USE_SPD=$1
    local USE_LIE=$2
    local USE_DIAG=$3
    local EXP_NAME=$4
    local SEED=$5

    echo "────────────────────────────────────────────────────────"
    echo "▶ LLM  ${EXP_NAME} | SEED=${SEED}"
    echo "  SPD=${USE_SPD} LIE=${USE_LIE} DIAG=${USE_DIAG} | w: ${LLM_W_INIT}→${LLM_W_FLOOR} over ${LLM_ANNEAL} steps"
    echo "────────────────────────────────────────────────────────"

    ARGS=(
        python3 src/train_fixed.py
        --env "$TASK"
        --run_name "$EXP_NAME"
        --total_timesteps "$STEPS"
        --horizon "$HORIZON"
        --n_envs "$N_ENVS"
        --gamma "$GAMMA"
        --gamma_start "$GAMMA_START"
        --num_markers "$NUM_MARKERS"
        --seed "$SEED"
        --lr "$LR"
        --batch_size "$BATCH_SIZE"
        --record_video
        --camera_names "frontview"
        --primitive_init "teleport" # Enable the teleport wrapper
        --use_quality_reward        # Checkpoint-gated quality reward (arXiv:2502.12599)
        --use_sequential_waypoints
        # LLM prior flags
        --use_llm_prior
        --llm_backend "$LLM_BACKEND"
        --llm_model "$LLM_MODEL"
        --llm_prior_weight "$LLM_W_INIT"
        --llm_anneal_steps "$LLM_ANNEAL"
        --llm_anneal_floor "$LLM_W_FLOOR"
        --llm_anneal_schedule "$LLM_ANNEAL_SCHEDULE"
        --llm_query_interval "$LLM_QUERY_INTERVAL"
        # profile auto-selected by train_fixed.py for Wipe → wipe_impedance_profile.yaml
    )
    [ "$USE_SPD" == "TRUE" ] && ARGS+=(--use_spd)
    [ "$USE_LIE" == "TRUE" ] && ARGS+=(--use_lie)
    [ "$USE_DIAG" == "TRUE" ] && ARGS+=(--use_diag)

    LOG="logs/wipe/${TASK}_${EXP_NAME}_SEED_${SEED}.log"
    echo "  → ${LOG}"
    "${ARGS[@]}" > "$LOG" 2>&1
    echo "  ✅ Done: ${EXP_NAME} SEED=${SEED}"
}

if [ "$LLM_BACKEND" == "ollama" ]; then
    echo "Starting Ollama server on port ${UNIQUE_PORT}..."
    ~/.local/bin/ollama serve &
    OLLAMA_PID=$!

    # Clean up Ollama on script exit, interruption, or job cancellation
    cleanup() {
        if [ -n "$OLLAMA_PID" ]; then
            echo "Stopping background Ollama server (PID $OLLAMA_PID)..."
            kill "$OLLAMA_PID" 2>/dev/null || true
        fi
    }
    trap cleanup EXIT INT TERM
    sleep 5

    echo "Pulling model ${LLM_MODEL} (if not already downloaded)..."
    ~/.local/bin/ollama pull ${LLM_MODEL}
    sleep 2

    echo ""
    echo "Checking Ollama availability at ${OLLAMA_HOST} ..."
    if ! check_ollama; then
        echo "❌ Ollama not responding. Aborting."
        exit 1
    fi
    echo "✅ Ollama OK — proceeding."
    echo "$OLLAMA_RESPONSE" | grep -o '"response":"[^"]*"'
    echo ""
else
    echo "Using OpenAI backend with model ${LLM_MODEL}. Bypassing local Ollama server."
    if [ -z "$OPENAI_API_KEY" ]; then
        echo "⚠️ WARNING: OPENAI_API_KEY is not set. Make sure it is exported in your environment!"
    fi
fi

TRAIN_PIDS=()

for SEED in "${SEEDS[@]}"; do
    echo "════════════════  SEED ${SEED}  ════════════════"

    # # BASELINE with LLM
    # wait_for_queue
    # run_llm "FALSE" "FALSE" "FALSE" "WIPE_ICRA_BASELINE_LLMv4_LLAMA_W${LLM_W_INIT}_LR${LR}_H${HORIZON}_LOGEUC_GEOBOUND" "$SEED" &
    # TRAIN_PIDS+=($!)

    # SPD with LLM
    wait_for_queue
    run_llm "TRUE"  "FALSE" "FALSE" "WIPE_ICRA_SPD_LLMv4_LLAMA_W${LLM_W_INIT}_LR${LR}_H${HORIZON}_LOGEUC_GEOBOUND" "$SEED" &
    TRAIN_PIDS+=($!)

done

# Wait ONLY for all RL training processes to finish
wait "${TRAIN_PIDS[@]}"

echo ""
echo "🎉  All TiltedWipe Teleport experiments complete!"

conda deactivate
