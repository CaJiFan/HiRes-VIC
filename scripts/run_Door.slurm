#!/bin/bash
#
# --- Slurm Configuration ---
#SBATCH --job-name=Door_HiResVIC
#SBATCH --output=logs/door_%j.txt
#SBATCH --error=logs/door_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --mem=160G                            
#SBATCH --time=168:00:00
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

mkdir -p logs/door

# ── Fixed hypers ──────────────────────────────────────────────────────────────
TASK="Door"
STEPS=1000000
HORIZON=50
N_ENVS=8
LR=3e-4
# GAMMA=0.90
GAMMA=$(awk "BEGIN {print 1 - 1/$HORIZON}")
SEEDS=(1 2 3)

# ── LLM prior schedule ────────────────────────────────────────────────────────
# Anneal from W_INIT → W_FLOOR over the first 25% of per-env steps.
# Per-env steps ≈ STEPS / N_ENVS = 250,000 / 8 = 31,250.
LLM_W_INIT=0.5
LLM_W_FLOOR=0.05
LLM_ANNEAL=$((STEPS * 25 / 100 / N_ENVS))   # = 31,250
LLM_BACKEND="ollama"
LLM_MODEL="llama3.2"
LLM_QUERY_INTERVAL=12

wait_for_queue() {
    while [ $(jobs -pr | wc -l) -ge 9 ]; do
        sleep 10
    done
}

# ==========================================
# 2. OLLAMA SERVER SETUP
# ==========================================
# Tell Ollama to save massive weights to your generous /home drive
export OLLAMA_MODELS="/home/cjimenez/ollama_models"

# Generate a unique port to avoid conflicts with other jobs on the same node
UNIQUE_PORT=$((11000 + (SLURM_JOB_ID % 1000)))
export OLLAMA_HOST="127.0.0.1:${UNIQUE_PORT}"

OLLAMA_RESPONSE=""
check_ollama() {
    OLLAMA_RESPONSE=$(curl -s -X POST "http://${OLLAMA_HOST}/api/generate" -d '{
        "model": "llama3.2:latest",
        "prompt": "Reply with one word: ready",
        "stream": false
    }' 2>/dev/null)
    echo "$OLLAMA_RESPONSE" | grep -q '"done":true'
}

# ── Helper functions ──────────────────────────────────────────────────────────
run_geo() {
    # Geometric-prior-only run (no LLM)
    local USE_SPD=$1   # "TRUE" | "FALSE"
    local USE_LIE=$2   # "TRUE" | "FALSE"
    local USE_DIAG=$3  # "TRUE" | "FALSE"
    local EXP_NAME=$4
    local SEED=$5

    echo "────────────────────────────────────────────────────────"
    echo "▶ GEO  ${EXP_NAME} | SEED=${SEED}"
    echo "  SPD=${USE_SPD} LIE=${USE_LIE} DIAG=${USE_DIAG}"
    echo "  steps=${STEPS} horizon=${HORIZON} n_envs=${N_ENVS} γ=${GAMMA}"
    echo "────────────────────────────────────────────────────────"

    ARGS=(
        python3 src/train_fixed.py
        --env "$TASK"
        --run_name "$EXP_NAME"
        --total_timesteps "$STEPS"
        --horizon "$HORIZON"
        --n_envs "$N_ENVS"
        --gamma "$GAMMA"
        --seed "$SEED"
        --lr "$LR"
        --record_video
        --camera_names "frontview"
        --primitive_init none
    )
    [ "$USE_SPD"  == "TRUE" ] && ARGS+=(--use_spd)
    [ "$USE_LIE"  == "TRUE" ] && ARGS+=(--use_lie)
    [ "$USE_DIAG" == "TRUE" ] && ARGS+=(--use_diag)

    LOG="logs/door/${TASK}_${EXP_NAME}_SEED_${SEED}.log"
    echo "  → ${LOG}"
    "${ARGS[@]}" > "$LOG" 2>&1
    echo "  ✅ Done: ${EXP_NAME} SEED=${SEED}"
}

run_llm() {
    # Geometric + LLM semantic prior run
    local USE_SPD=$1
    local USE_LIE=$2
    local EXP_NAME=$3
    local SEED=$4

    echo "────────────────────────────────────────────────────────"
    echo "▶ LLM  ${EXP_NAME} | SEED=${SEED}"
    echo "  SPD=${USE_SPD} LIE=${USE_LIE} | w: ${LLM_W_INIT}→${LLM_W_FLOOR} over ${LLM_ANNEAL} steps"
    echo "────────────────────────────────────────────────────────"

    ARGS=(
        python3 src/train_fixed.py
        --env "$TASK"
        --run_name "$EXP_NAME"
        --total_timesteps "$STEPS"
        --horizon "$HORIZON"
        --n_envs "$N_ENVS"
        --gamma "$GAMMA"
        --seed "$SEED"
        --lr "$LR"
        --record_video
        --camera_names "frontview"
        --primitive_init none
        # LLM prior flags
        --use_llm_prior
        --llm_backend "$LLM_BACKEND"
        --llm_model "$LLM_MODEL"
        --llm_prior_weight "$LLM_W_INIT"
        --llm_anneal_steps "$LLM_ANNEAL"
        --llm_anneal_floor "$LLM_W_FLOOR"
        --llm_query_interval "$LLM_QUERY_INTERVAL"
        # profile auto-selected by train_fixed.py for Door
    )
    [ "$USE_SPD" == "TRUE" ] && ARGS+=(--use_spd)
    [ "$USE_LIE" == "TRUE" ] && ARGS+=(--use_lie)

    LOG="logs/door/${TASK}_${EXP_NAME}_SEED_${SEED}.log"
    echo "  → ${LOG}"
    "${ARGS[@]}" > "$LOG" 2>&1
    echo "  ✅ Done: ${EXP_NAME} SEED=${SEED}"
}

echo "Starting Ollama server on port ${UNIQUE_PORT}..."
~/.local/bin/ollama serve &

# Wait for the server to fully initialize
sleep 10

echo ""
echo "Checking Ollama availability at ${OLLAMA_HOST} ..."
if ! check_ollama; then
    echo "❌ Ollama not responding. Aborting."
    exit 1
fi
echo "✅ Ollama OK — proceeding."
echo "$OLLAMA_RESPONSE" | grep -o '"response":"[^"]*"'
echo ""

for SEED in "${SEEDS[@]}"; do
    echo "════════════════  SEED ${SEED}  ════════════════"

    # 1. BASELINE — diagonal variable_kp, Euclidean obs
    wait_for_queue
    run_geo "FALSE" "FALSE" "FALSE" "BASELINE_FINAL_LR${LR}_H${HORIZON}_G${GAMMA}" "$SEED" &

    # 2. DIAG — diagonal SPD (log-scale exponential map)
    wait_for_queue
    run_geo "FALSE" "FALSE" "TRUE"  "DIAG_FINAL_LR${LR}_H${HORIZON}_G${GAMMA}" "$SEED" &

    # 3. SPD_ONLY — full 3×3 SPD manifold
    wait_for_queue
    run_geo "TRUE"  "FALSE" "FALSE" "SPD_ONLY_FINAL_LR${LR}_H${HORIZON}_G${GAMMA}" "$SEED" &
    
    # 4. LIE — SO(3) Lie-group observation only
    wait_for_queue
    run_geo "FALSE" "TRUE"  "FALSE" "LIE_ONLY_FINAL_LR${LR}_H${HORIZON}_G${GAMMA}" "$SEED" &

    # 5. FULL_GRL — full SPD + SO(3) Lie-group observation
    wait_for_queue
    run_geo "TRUE"  "TRUE"  "FALSE" "FULL_GRL_FINAL_LR${LR}_H${HORIZON}_G${GAMMA}" "$SEED" &

    # 6. BASELINE with LLM
    wait_for_queue
    run_llm "FALSE" "FALSE" "BASELINE_FINAL_LR${LR}_H${HORIZON}_LLM_W${LLM_W_INIT}_G${GAMMA}" "$SEED" &

    # 7. SPD with LLM
    wait_for_queue
    run_llm "TRUE"  "FALSE" "SPD_ONLY_FINAL_LR${LR}_H${HORIZON}_LLM_W${LLM_W_INIT}_G${GAMMA}" "$SEED" &

    # 9. FULL_GRL + LLM
    wait_for_queue
    run_llm "TRUE"  "TRUE"  "FULL_GRL_FINAL_LR${LR}_H${HORIZON}_LLM_W${LLM_W_INIT}_G${GAMMA}" "$SEED" &

done

echo ""
echo "🎉  All Door experiments complete!"