#!/bin/bash
#
# --- Slurm Configuration ---
#SBATCH --job-name=Wipe_HiResVIC
#SBATCH --output=logs/wipe_%j.txt
#SBATCH --error=logs/wipe_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=160G
#SBATCH --time=168:00:00
#SBATCH --gres=gpu:lovelace:1

# =============================================================================
# Run TiltedWipe environment experiments (HiRes-VIC thesis)
#
# Ablation matrix:
#   Section A — Geometric prior only (no LLM)
#     BASELINE   — diagonal variable_kp (linear scale)
#     DIAG       — diagonal SPD (log-scale exponential map)
#     SPD_ONLY   — full 3×3 SPD manifold (Mandel basis)
#     LIE_ONLY   — SO(3) log-map observation only
#     FULL_GRL   — full SPD + SO(3) log-map observation
#
#   Section B — Geometric + LLM semantic prior (SPD configs only)
#     SPD_LLM    — SPD manifold + LLM impedance prior (annealed)
#     FULL_LLM   — SPD + SO(3) + LLM impedance prior (annealed)
#
# LLM setup:
#   - Ollama must be running locally (ollama serve) with llama3.2 pulled
#   - Profile: configs/wipe_impedance_profile.yaml (auto-selected by train_fixed.py)
#   - w schedule: prior_weight=0.7 → anneal_floor=0.05 over 75% of training
#
# Usage:
#   sbatch scripts/run_TiltedWipe.slurm
# =============================================================================

set -e

eval "$(conda shell.bash hook)"
conda activate tfm

mkdir -p logs/wipe

# --- Admin Required L40S Fixes ---
export NCCL_P2P_DISABLE=1
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024
export MALLOC_ARENA_MAX=1

# ── Fixed hypers ──────────────────────────────────────────────────────────────
TASK="TiltedWipe"
STEPS=2000000
HORIZON=200
N_ENVS=8
# GAMMA=0.95
GAMMA=$(awk "BEGIN {print 1 - 1/$HORIZON}")
NUM_MARKERS=100
SEEDS=(1 2 3)

# ── LLM prior schedule ────────────────────────────────────────────────────────
# Anneal from W_INIT → W_FLOOR over the first 75% of per-env steps.
# Per-env steps ≈ STEPS / N_ENVS = 125,000. 75% of that = 93,750.
LLM_W_INIT=0.7
LLM_W_FLOOR=0.4
LLM_ANNEAL=$((STEPS * 75 / 100 / N_ENVS))   # = 93,750
LLM_BACKEND="ollama"
LLM_MODEL="llama3.2"
LLM_QUERY_INTERVAL=50

wait_for_queue() {
    while [ $(jobs -pr | wc -l) -ge 9 ]; do
        sleep 10
    done
}

# ── OLLAMA SERVER SETUP ───────────────────────────────────────────────────────
export OLLAMA_MODELS="/home/cjimenez/ollama_models"
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
    echo "  steps=${STEPS} horizon=${HORIZON} n_envs=${N_ENVS} γ=${GAMMA} markers=${NUM_MARKERS}"
    echo "────────────────────────────────────────────────────────"

    ARGS=(
        python3 src/train_fixed.py
        --env "$TASK"
        --run_name "$EXP_NAME"
        --total_timesteps "$STEPS"
        --horizon "$HORIZON"
        --n_envs "$N_ENVS"
        --gamma "$GAMMA"
        --num_markers "$NUM_MARKERS"
        --seed "$SEED"
        --record_video
        --camera_names "frontview,robot0_eye_in_hand"
    )
    [ "$USE_SPD"  == "TRUE" ] && ARGS+=(--use_spd)
    [ "$USE_LIE"  == "TRUE" ] && ARGS+=(--use_lie)
    [ "$USE_DIAG" == "TRUE" ] && ARGS+=(--use_diag)

    LOG="logs/wipe/${TASK}_${EXP_NAME}_SEED_${SEED}.log"
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
        --num_markers "$NUM_MARKERS"
        --seed "$SEED"
        --record_video
        --camera_names "frontview"
        # LLM prior flags
        --use_llm_prior
        --llm_backend "$LLM_BACKEND"
        --llm_model "$LLM_MODEL"
        --llm_prior_weight "$LLM_W_INIT"
        --llm_anneal_steps "$LLM_ANNEAL"
        --llm_anneal_floor "$LLM_W_FLOOR"
        --llm_query_interval "$LLM_QUERY_INTERVAL"
        # profile auto-selected by train_fixed.py for Wipe → wipe_impedance_profile.yaml
    )
    [ "$USE_SPD" == "TRUE" ] && ARGS+=(--use_spd)
    [ "$USE_LIE" == "TRUE" ] && ARGS+=(--use_lie)

    LOG="logs/wipe/${TASK}_${EXP_NAME}_SEED_${SEED}.log"
    echo "  → ${LOG}"
    "${ARGS[@]}" > "$LOG" 2>&1
    echo "  ✅ Done: ${EXP_NAME} SEED=${SEED}"
}

echo "Starting Ollama server on port ${UNIQUE_PORT}..."
~/.local/bin/ollama serve &
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
    run_geo "FALSE" "FALSE" "FALSE" "BASELINE_H${HORIZON}_G${GAMMA}" "$SEED" &

    # 2. DIAG — diagonal SPD (log-scale exponential map)
    wait_for_queue
    run_geo "FALSE" "FALSE" "TRUE"  "DIAG_H${HORIZON}_G${GAMMA}" "$SEED" &

    # 3. SPD_ONLY — full 3×3 SPD manifold
    wait_for_queue
    run_geo "TRUE"  "FALSE" "FALSE" "SPD_ONLY_H${HORIZON}_G${GAMMA}" "$SEED" &

    # 4. LIE_ONLY — SO(3) Lie-group observation only
    wait_for_queue
    run_geo "FALSE" "TRUE"  "FALSE" "LIE_ONLY_H${HORIZON}_G${GAMMA}" "$SEED" &

    # 5. FULL_GRL — full SPD + SO(3) Lie-group observation
    wait_for_queue
    run_geo "TRUE"  "TRUE"  "FALSE" "FULL_GRL_H${HORIZON}_G${GAMMA}" "$SEED" &

    # 6. BASELINE with LLM
    wait_for_queue
    run_llm "FALSE"  "FALSE" "BASELINE_H${HORIZON}_LLM_W${LLM_W_INIT}_G${GAMMA}" "$SEED" &

    # 7. SPD with LLM
    wait_for_queue
    run_llm "TRUE"  "FALSE" "SPD_ONLY_H${HORIZON}_LLM_W${LLM_W_INIT}_G${GAMMA}" "$SEED" &

    # 8. LIE with LLM
    wait_for_queue
    run_llm "FALSE" "TRUE"  "LIE_ONLY_H${HORIZON}_LLM_W${LLM_W_INIT}_G${GAMMA}" "$SEED" &

    # 9. FULL_GRL + LLM
    wait_for_queue
    run_llm "TRUE"  "TRUE"  "FULL_GRL_H${HORIZON}_LLM_W${LLM_W_INIT}_G${GAMMA}" "$SEED" &

done

wait

echo ""
echo "🎉  All TiltedWipe experiments complete!"

conda deactivate
