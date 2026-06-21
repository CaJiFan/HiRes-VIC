#!/bin/bash
#
# --- Slurm Configuration ---
#SBATCH --job-name=NutSquare_HiResVIC
#SBATCH --output=logs/nutsquare_%j.txt
#SBATCH --error=logs/nutsquare_%j.err
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

mkdir -p logs/nut_square

export NCCL_P2P_DISABLE=1
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024
export MALLOC_ARENA_MAX=1

TASK="NutAssemblySquare"
STEPS=1000000
HORIZON=80
N_ENVS=8
GAMMA=0.90
# GAMMA=$(awk "BEGIN {print 1 - 1/$HORIZON}")
SEEDS=(1 2 3)

LLM_W_INIT=0.8
LLM_W_FLOOR=0.05
LLM_ANNEAL=$((STEPS * 25 / 100 / N_ENVS))
LLM_BACKEND="ollama"
LLM_MODEL="llama3.2"
LLM_QUERY_INTERVAL=20

wait_for_queue() {
    while [ $(jobs -pr | wc -l) -ge 9 ]; do
        sleep 10
    done
}

export OLLAMA_MODELS="/home/cjimenez/ollama_models"
UNIQUE_PORT=$((11000 + (SLURM_JOB_ID % 1000)))
export OLLAMA_HOST="127.0.0.1:${UNIQUE_PORT}"

echo "Starting Ollama server on port ${UNIQUE_PORT}..."
~/.local/bin/ollama serve &
sleep 10

check_ollama() {
    OLLAMA_RESPONSE=$(curl -s -X POST "http://${OLLAMA_HOST}/api/generate" -d '{
        "model": "llama3.2:latest",
        "prompt": "Reply with one word: ready",
        "stream": false
    }' 2>/dev/null)
    echo "$OLLAMA_RESPONSE" | grep -q '"done":true'
}

run_geo() {
    local USE_SPD=$1
    local USE_LIE=$2
    local USE_DIAG=$3
    local EXP_NAME=$4
    local SEED=$5

    ARGS=(
        python3 src/train_fixed.py
        --env "$TASK"
        --run_name "$EXP_NAME"
        --total_timesteps "$STEPS"
        --horizon "$HORIZON"
        --n_envs "$N_ENVS"
        --gamma "$GAMMA"
        --seed "$SEED"
        --record_video
        --camera_names "frontview,robot0_eye_in_hand"
        --primitive_init teleport
    )
    [ "$USE_SPD"  == "TRUE" ] && ARGS+=(--use_spd)
    [ "$USE_LIE"  == "TRUE" ] && ARGS+=(--use_lie)
    [ "$USE_DIAG" == "TRUE" ] && ARGS+=(--use_diag)

    LOG="logs/nut_square/${TASK}_${EXP_NAME}_SEED_${SEED}.log"
    "${ARGS[@]}" > "$LOG" 2>&1
    echo "  ✅ Done: ${EXP_NAME} SEED=${SEED}"
}

run_llm() {
    local USE_SPD=$1
    local USE_LIE=$2
    local EXP_NAME=$3
    local SEED=$4

    ARGS=(
        python3 src/train_fixed.py
        --env "$TASK"
        --run_name "$EXP_NAME"
        --total_timesteps "$STEPS"
        --horizon "$HORIZON"
        --n_envs "$N_ENVS"
        --gamma "$GAMMA"
        --seed "$SEED"
        --record_video
        --camera_names "frontview,robot0_eye_in_hand"
        --primitive_init teleport
        --use_llm_prior
        --llm_backend "$LLM_BACKEND"
        --llm_model "$LLM_MODEL"
        --llm_prior_weight "$LLM_W_INIT"
        --llm_anneal_steps "$LLM_ANNEAL"
        --llm_anneal_floor "$LLM_W_FLOOR"
        --llm_query_interval "$LLM_QUERY_INTERVAL"
    )
    [ "$USE_SPD" == "TRUE" ] && ARGS+=(--use_spd)
    [ "$USE_LIE" == "TRUE" ] && ARGS+=(--use_lie)

    LOG="logs/nut_square/${TASK}_${EXP_NAME}_SEED_${SEED}.log"
    "${ARGS[@]}" > "$LOG" 2>&1
    echo "  ✅ Done: ${EXP_NAME} SEED=${SEED}"
}

echo "Checking Ollama availability at ${OLLAMA_HOST} ..."
if ! check_ollama; then
    echo "❌ Ollama not responding. Aborting."
    exit 1
fi
echo "✅ Ollama OK — proceeding."

for SEED in "${SEEDS[@]}"; do
    echo "════════════════  SEED ${SEED}  ════════════════"

    wait_for_queue
    run_geo "FALSE" "FALSE" "FALSE" "BASELINE_H${HORIZON}_G${GAMMA}" "$SEED" &

    wait_for_queue
    run_geo "FALSE" "FALSE" "TRUE"  "DIAG_H${HORIZON}_G${GAMMA}" "$SEED" &

    wait_for_queue
    run_geo "TRUE"  "FALSE" "FALSE" "SPD_ONLY_H${HORIZON}_G${GAMMA}" "$SEED" &

    wait_for_queue
    run_geo "FALSE" "TRUE"  "FALSE" "LIE_ONLY_H${HORIZON}_G${GAMMA}" "$SEED" &

    wait_for_queue
    run_geo "TRUE"  "TRUE"  "FALSE" "FULL_GRL_H${HORIZON}_G${GAMMA}" "$SEED" &

    wait_for_queue
    run_llm "FALSE"  "FALSE" "BASELINE_H${HORIZON}_LLM_W${LLM_W_INIT}_G${GAMMA}" "$SEED" &

    wait_for_queue
    run_llm "TRUE"  "FALSE" "SPD_ONLY_H${HORIZON}_LLM_W${LLM_W_INIT}_G${GAMMA}" "$SEED" &

    wait_for_queue
    run_llm "TRUE"  "TRUE"  "FULL_GRL_H${HORIZON}_LLM_W${LLM_W_INIT}_G${GAMMA}" "$SEED" &
    
    # Wait for all 9 jobs of the current seed to finish before moving to the next seed
    # wait
done

echo "��  All NutAssemblySquare experiments complete!"
conda deactivate
