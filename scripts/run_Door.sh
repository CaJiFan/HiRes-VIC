#!/bin/bash
# =============================================================================
# Run Door environment experiments (HiRes-VIC thesis)
#
# Ablation matrix:
#   Section A — Geometric prior only (no LLM)
#     BASELINE   — diagonal variable_kp (linear scale)
#     DIAG       — diagonal SPD (log-scale exponential map)
#     SPD_ONLY   — full 3×3 SPD manifold (Mandel basis)
#     FULL_GRL   — full SPD + SO(3) log-map observation
#
#   Section B — Geometric + LLM semantic prior (SPD configs only)
#     SPD_LLM    — SPD manifold + LLM impedance prior (annealed)
#     FULL_LLM   — SPD + SO(3) + LLM impedance prior (annealed)
#
# LLM setup:
#   - Ollama must be running locally (ollama serve) with llama3.2 pulled
#   - Profile: configs/door_impedance_profile.yaml (auto-selected by train_fixed.py)
#   - w schedule: prior_weight=0.7 → anneal_floor=0.05 over 75% of training
#
# Usage:
#   bash scripts/run_Door.sh            # sequential
#   nohup bash scripts/run_Door.sh &    # background
# =============================================================================

set -e

eval "$(conda shell.bash hook)"
conda activate tfm

mkdir -p logs/door

# ── Fixed hypers ──────────────────────────────────────────────────────────────
TASK="Door"
STEPS=1000000
HORIZON=100
N_ENVS=16
LR=3e-4
BATCH_SIZE=1024
GAMMA=0.95
GAMMA_START=0.95 
SEEDS=(0 1 2 3 4)

# ── LLM prior schedule ────────────────────────────────────────────────────────
# Anneal from W_INIT → W_FLOOR over the first 75% of per-env steps.
# Per-env steps ≈ STEPS / N_ENVS = 125,000. 75% of that = 93,750.
LLM_W_INIT=0.4
LLM_W_FLOOR=$LLM_W_INIT
# LLM_ANNEAL=$((STEPS * 75 / 100 / N_ENVS))   # = 93,750
LLM_ANNEAL=0
LLM_BACKEND="ollama"
LLM_MODEL="llama3.2"
LLM_QUERY_INTERVAL=50

# ── Ollama smoke test (only needed if running LLM configs) ────────────────────
OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
export OLLAMA_HOST

check_ollama() {
    RESPONSE=$(curl -s -X POST "http://${OLLAMA_HOST}/api/generate" -d '{
        "model": "llama3.2:latest",
        "prompt": "Reply with one word: ready",
        "stream": false
    }' 2>/dev/null)
    echo "$RESPONSE" | grep -q '"done":true'
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
        --lr "$LR"
        --batch_size "$BATCH_SIZE"
        --gamma "$GAMMA"
        --gamma_start "$GAMMA_START"
        --seed "$SEED"
        --camera_names "frontview"
        --primitive_init none
        --record_video
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
        --camera_names "frontview,robot0_eye_in_hand"
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

# ── Section A: Geometric prior sweep ─────────────────────────────────────────
echo ""
echo "██████████████  SECTION A: Geometric Prior  ██████████████"

for SEED in "${SEEDS[@]}"; do
    echo ""
    echo "════════════════  SEED ${SEED}  ════════════════"

    # 1. BASELINE — diagonal variable_kp, Euclidean obs
    run_geo "FALSE" "FALSE" "FALSE" "DOOR_BASELINE_G${GAMMA}" "$SEED"

    # 2. DIAG — diagonal SPD (log-scale exponential map)
    run_geo "FALSE" "FALSE" "TRUE"  "DOOR_DIAG_G${GAMMA}" "$SEED"

    # 3. SPD_ONLY — full 3×3 SPD manifold
    run_geo "TRUE"  "FALSE" "FALSE" "DOOR_SPD_ONLY_G${GAMMA}" "$SEED"

    # 4. FULL_GRL — full SPD + SO(3) Lie-group observation
    run_geo "TRUE"  "TRUE"  "FALSE" "DOOR_FULL_GRL_G${GAMMA}" "$SEED"
done

# # ── Section B: Geometric + LLM prior ─────────────────────────────────────────
# echo ""
# echo "██████████████  SECTION B: Geometric + LLM Prior  ██████████████"
# echo ""
# echo "Checking Ollama availability at ${OLLAMA_HOST} ..."
# if ! check_ollama; then
#     echo "❌ Ollama not responding. Skipping Section B."
#     echo "   To run LLM configs later: bash scripts/run_Door_llm_only.sh"
#     echo ""
# else
#     echo "✅ Ollama OK — proceeding with LLM runs."
#     echo ""

#     for SEED in "${SEEDS[@]}"; do
#         echo ""
#         echo "════════════════  SEED ${SEED}  ════════════════"

#         # 5. SPD + LLM — SPD manifold with annealed semantic prior
#         run_llm "TRUE"  "FALSE" "DOOR_SPD_LLM_W${LLM_W_INIT}_G${GAMMA}" "$SEED"

#         # 6. FULL_GRL + LLM — SPD + SO(3) + LLM prior
#         run_llm "TRUE"  "TRUE"  "DOOR_FULL_LLM_W${LLM_W_INIT}_G${GAMMA}" "$SEED"
#     done
# fi

echo ""
echo "🎉  All Door experiments complete!"