#!/bin/bash
#
# --- Slurm Configuration ---
#SBATCH --job-name=Gamma_Search
#SBATCH --output=logs/gamma_%j.txt
#SBATCH --error=logs/gamma_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=100G                            
#SBATCH --time=48:00:00
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

mkdir -p logs/gamma_search

# ── Fixed hypers ──────────────────────────────────────────────────────────────
N_ENVS=8
SEEDS=(1 2 3)


wait_for_queue() {
    while [ $(jobs -pr | wc -l) -ge 4 ]; do
        sleep 10
    done
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
        --record_video
    )

    [ "$TASK" == "NutAssemblySquare" ] && ARGS+=(--primitive_init teleport)
    [ "$TASK" == "NutAssemblySquare" ] && ARGS+=(--camera_names "frontview,robot0_eye_in_hand")
    [ "$TASK" != "NutAssemblySquare" ] && ARGS+=(--camera_names "frontview")
    [ "$TASK" == "TiltedWipe" ] && ARGS+=(--num_markers "${NUM_MARKERS}")
    [ "$USE_SPD"  == "TRUE" ] && ARGS+=(--use_spd)
    [ "$USE_LIE"  == "TRUE" ] && ARGS+=(--use_lie)
    [ "$USE_DIAG" == "TRUE" ] && ARGS+=(--use_diag)

    LOG="logs/gamma_search/${TASK}/${TASK}_${EXP_NAME}_SEED_${SEED}.log"
    echo "  → ${LOG}"
    "${ARGS[@]}" > "$LOG" 2>&1
    echo "  ✅ Done: ${EXP_NAME} SEED=${SEED}"
}



# TASK="Door"
# STEPS=1500000
# HORIZON=50
# EFFECTIVE_GAMMA=$(awk "BEGIN {print 1 - 1/$HORIZON}")
# mkdir -p "logs/gamma_search/${TASK}"
# for SEED in "${SEEDS[@]}"; do
#     echo "════════════════  SEED ${SEED}  ════════════════"
#     for GAMMA in "$EFFECTIVE_GAMMA" 0.95 0.90; do
#         echo "Running for GAMMA = $GAMMA"

        
#         wait_for_queue
#         run_geo "TRUE"  "TRUE"  "FALSE" "FULL_GRL_FINAL_H${HORIZON}_G${GAMMA}" "$SEED" &
#     done

# done



# TASK="NutAssemblySquare"
# STEPS=1000000
# HORIZON=80
# EFFECTIVE_GAMMA=$(awk "BEGIN {print 1 - 1/$HORIZON}")
# mkdir -p "logs/gamma_search/${TASK}"
# for SEED in "${SEEDS[@]}"; do
#     echo "════════════════  SEED ${SEED}  ════════════════"
#     for GAMMA in "$EFFECTIVE_GAMMA" 0.95 0.90; do
#         echo "Running for GAMMA = $GAMMA"
    
#         wait_for_queue
#         run_geo "TRUE"  "TRUE"  "FALSE" "FULL_GRL_FINAL_H${HORIZON}_G${GAMMA}" "$SEED" &
#     done
# done


TASK="TiltedWipe"
STEPS=1000000
HORIZON=150
EFFECTIVE_GAMMA=$(awk "BEGIN {print 1 - 1/$HORIZON}")
NUM_MARKERS=50
mkdir -p "logs/gamma_search/${TASK}"
for SEED in "${SEEDS[@]}"; do
    echo "════════════════  SEED ${SEED}  ════════════════"
    for GAMMA in "$EFFECTIVE_GAMMA" 0.95 0.90; do
        echo "Running for GAMMA = $GAMMA"
        
        wait_for_queue
        run_geo "TRUE"  "TRUE"  "FALSE" "FULL_GRL_FINAL_H${HORIZON}_G${GAMMA}" "$SEED" &
    done
done

# Wait for the final batch of jobs to finish before exiting
wait

echo ""
echo "🎉  All experiments complete!"