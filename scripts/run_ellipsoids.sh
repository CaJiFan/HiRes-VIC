#!/bin/bash
# ============================================================
# run_ellipsoids.sh — Regenerate stiffness ellipsoid plots
# for BASELINE vs SPD_ONLY across all three environments.
# Picks the best available seed (highest numbered with a model).
# ============================================================

set -e
BEST_MODELS="logs/best_models"
OUTPUT_DIR="outputs"
mkdir -p "$OUTPUT_DIR"

export PYTHONPATH=/home/cjimenez/projects/HiRes-VIC:$PYTHONPATH
export MUJOCO_GL="egl"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

find_best_model() {
    local prefix="$1"
    # Search for seeds 3, 2, 1 in order (prefer higher seeds for consistency with paper)
    for seed in 3 2 1; do
        local path="$BEST_MODELS/${prefix}_SEED_${seed}/best_model.zip"
        if [ -f "$path" ]; then
            echo "$path"
            return 0
        fi
    done
    # Try without SEED suffix (some older runs)
    local path="$BEST_MODELS/${prefix}/best_model.zip"
    if [ -f "$path" ]; then
        echo "$path"
        return 0
    fi
    echo ""
}

run_ellipsoid() {
    local env="$1"
    local baseline_prefix="$2"
    local spd_prefix="$3"

    echo ""
    echo "============================================================"
    echo "  Generating ellipsoids for: $env"
    echo "============================================================"

    baseline_path=$(find_best_model "$baseline_prefix")
    spd_path=$(find_best_model "$spd_prefix")

    if [ -z "$baseline_path" ]; then
        echo "  ❌ BASELINE model not found for prefix: $baseline_prefix — skipping"
        return 1
    fi
    if [ -z "$spd_path" ]; then
        echo "  ❌ SPD_ONLY model not found for prefix: $spd_prefix — skipping"
        return 1
    fi

    echo "  BASELINE: $baseline_path"
    echo "  SPD_ONLY: $spd_path"

    python3 scripts/plot_ellipsoids.py \
        --env "$env" \
        --baseline_path "$baseline_path" \
        --spd_path "$spd_path"

    echo "  ✅ Done: outputs/${env}_ellipsoids.png"
}

# ── TiltedWipe ────────────────────────────────────────────────
run_ellipsoid \
    "TiltedWipe" \
    "SAC_TILTEDWIPE_BASELINE_FINAL_H150_G0.95" \
    "SAC_TILTEDWIPE_SPD_ONLY_FINAL_H150_G0.95"

# ── Door ──────────────────────────────────────────────────────
run_ellipsoid \
    "Door" \
    "SAC_DOOR_BASELINE_H50_G0.95" \
    "SAC_DOOR_SPD_ONLY_H50_G0.95"

# ── NutAssemblySquare ─────────────────────────────────────────
run_ellipsoid \
    "NutAssemblySquare" \
    "SAC_NUTASSEMBLYSQUARE_BASELINE_H80_G0.90" \
    "SAC_NUTASSEMBLYSQUARE_SPD_ONLY_H80_G0.90"

echo ""
echo "All ellipsoid plots saved to $OUTPUT_DIR/"
echo "Copy to manuscript/images/ with:"
echo "  cp outputs/*_ellipsoids.png manuscript/images/"
