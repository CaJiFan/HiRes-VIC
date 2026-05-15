#!/bin/bash
# =============================================================================
# LLM-Prior Hyperparameter Sweep — HiRes-VIC
# =============================================================================
#
# Sweeps the two most impactful LLM-RL integration hyperparameters:
#
#   --llm_prior_weight W    (blend factor)
#       Controls how strongly the LLM prior biases SAC's stiffness output.
#       ã[:9] = (1-W)*π_θ(s)[:9] + W*prior
#       Too high → LLM dominates, SAC loses exploration freedom in stiffness space.
#       Too low  → LLM has negligible effect; equivalent to no prior.
#       Sweep: 0.1, 0.2, 0.4
#
#   --llm_query_interval K  (steps between LLM queries)
#       How many env steps between async LLM calls.
#       Too low  → queries faster than phase transitions occur (wasted calls, stale).
#       Too high → LLM misses contact/wipe phase changes; prior is always stale.
#       At ~20Hz control, 1s ≈ 20 steps. Reasonable phase duration ~2-10s.
#       Sweep: 20, 50, 100
#
#   --gamma G=0.90           (fixed — established as optimal from baseline runs)
#       Re-sweeping gamma is unnecessary: the LLM prior only affects the stiffness
#       subspace, not the reward horizon, so optimal gamma is unchanged.
#
# Run order: baseline first, then grid. Runs are sequential (shared GPU + Ollama).
# Each run writes its own log to logs/sweep_llm/.
#
# Usage:
#   bash scripts/sweep_llm_hparam.sh                            # ollama (default)
#   LLM_BACKEND=openai bash scripts/sweep_llm_hparam.sh         # OpenAI
#   bash scripts/sweep_llm_hparam.sh 300000                     # shorter run
# =============================================================================

set -e

# ── Config ────────────────────────────────────────────────────────────────────
STEPS=${1:-500000}          # override with first arg, e.g. --steps 300000
SEED=3
N_ENVS=4                    # keep low: shared LLM server handles sequential requests better
ENV="TiltedWipe"
ALGO="SAC"
LOG_DIR="logs/sweep_llm"
GAMMA_FIXED=0.90            # established as optimal from baseline SPD runs
LLM_BACKEND=${LLM_BACKEND:-ollama}  # override: LLM_BACKEND=openai bash sweep_llm_hparam.sh

# Sweep values
PRIOR_WEIGHTS=(0.1 0.2 0.4)
QUERY_INTERVALS=(20 50 100)

mkdir -p "$LOG_DIR"

# ── Helper ────────────────────────────────────────────────────────────────────
run_train() {
    local RUN_NAME=$1
    shift
    local EXTRA_ARGS=("$@")

    local LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"

    echo ""
    echo "══════════════════════════════════════════════"
    echo " Run : $RUN_NAME"
    echo " Args: ${EXTRA_ARGS[*]}"
    echo " Log : $LOG_FILE"
    echo " Time: $(date)"
    echo "══════════════════════════════════════════════"

    python3 src/train_fixed.py \
        --env        "$ENV"   \
        --algorithm  "$ALGO"  \
        --total_timesteps "$STEPS" \
        --n_envs     "$N_ENVS" \
        --seed       "$SEED"  \
        --gamma      "$GAMMA_FIXED" \
        --run_name   "$RUN_NAME" \
        --use_spd \
        "${EXTRA_ARGS[@]}" \
        > "$LOG_FILE" 2>&1

    echo "✓ Done: $RUN_NAME"
}

# ── 0. Baseline (no LLM) ─────────────────────────────────────────────────────
# Always run first — establishes the SPD-only ceiling to compare against.


# ── 1. Main grid: prior_weight × query_interval ───────────────────────────────
# This is the primary sweep. Fixes gamma=GAMMA_FIXED, varies W and K.
# Expected insight:
#   - W too high (0.4) + K too low (20): LLM overwhelms policy, poor exploration
#   - W too low (0.1) + K too high (100): LLM barely visible, close to baseline
#   - Sweet spot likely around W=0.2, K=50 for a 300-step Wipe episode
echo ""
echo "▶ Starting main grid: prior_weight × query_interval"

for W in "${PRIOR_WEIGHTS[@]}"; do
    for K in "${QUERY_INTERVALS[@]}"; do
        RUN="SPD_LLM_w${W}_k${K}_g${GAMMA_FIXED}"
        run_train "$RUN" \
            --use_llm_prior \
            --llm_backend      "$LLM_BACKEND" \
            --llm_prior_weight "$W" \
            --llm_query_interval "$K"
    done
done

run_train "SPD_baseline_g${GAMMA_FIXED}"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════"
echo " All sweep runs completed."
echo " Total runs: $((1 + ${#PRIOR_WEIGHTS[@]} * ${#QUERY_INTERVALS[@]}))"
echo " Logs in: $LOG_DIR/"
echo " Compare runs in W&B project: HiRes-VIC"
echo "══════════════════════════════════════════════"
