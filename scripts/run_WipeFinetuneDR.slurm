#!/bin/bash
#
# --- Slurm Configuration ---
#SBATCH --job-name=wipe_finetune_dr
#SBATCH --output=logs/wipe_finetune_dr_%j.txt
#SBATCH --error=logs/wipe_finetune_dr_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=12:00:00
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

MAX_CONCURRENT=2
CURRENT_JOBS=0

wait_for_queue() {
    while [ $CURRENT_JOBS -ge $MAX_CONCURRENT ]; do
        wait -n
        CURRENT_JOBS=$((CURRENT_JOBS - 1))
    done
}

# --- Base Training Params ---
HORIZON=150
NUM_MARKERS=50
LR="5e-5"
BATCH_SIZE=1024
GAMMA=$(awk "BEGIN {print 1 - 1/$HORIZON}")
TOTAL_TIMESTEPS=1500000

# --- Fine-tune Function ---
run_finetune() {
    local USE_SPD=$1
    local USE_LIE=$2
    local USE_DIAG=$3
    local OLD_RUN_NAME=$4
    local SEED=$5

    # Path to the best model to resume from
    local LOAD_MODEL="logs/best_models/${OLD_RUN_NAME}_SEED_${SEED}/best_model.zip"
    
    # New run name for logging
    local NEW_RUN_NAME="${OLD_RUN_NAME}_DOMAINRAND"

    local ARGS="--env TiltedWipe --run_name ${NEW_RUN_NAME} --seed ${SEED} "
    ARGS+="--total_timesteps ${TOTAL_TIMESTEPS} --batch_size ${BATCH_SIZE} "
    ARGS+="--gamma ${GAMMA} --horizon ${HORIZON} --num_markers ${NUM_MARKERS} "
    ARGS+="--primitive_init teleport --n_envs 16 --lr ${LR} "
    
    # Enable Domain Randomization for fine-tuning
    ARGS+="--use_domain_rand "

    
    
    # Enable Universal Stiffness EMA Filter for smoothing
    ARGS+="--use_ema "

    ARGS+="--record_video "
    
    # Load the best checkpoint to resume
    ARGS+="--load_model ${LOAD_MODEL} "

    if [ "$USE_SPD" = "TRUE" ]; then ARGS+="--use_spd "; fi
    if [ "$USE_LIE" = "TRUE" ]; then ARGS+="--use_lie "; fi
    if [ "$USE_DIAG" = "TRUE" ]; then ARGS+="--use_diag "; fi

    echo "▶️ Launching fine-tuning: ${NEW_RUN_NAME} (Seed: ${SEED})"
    echo "  Loading from: ${LOAD_MODEL}"
    
    python src/train_fixed.py $ARGS > "logs/wipe/finetune_${NEW_RUN_NAME}_SEED_${SEED}.txt" 2>&1
}

echo "Starting Domain Randomization Fine-Tuning..."

# 1. BASELINE Run (Best Seed from CSV: 4)
wait_for_queue
run_finetune "FALSE" "FALSE" "FALSE" "SAC_WIPE_ICRA_BASELINE_LR1e-4_H150_CURRICULUM16" "4" &
CURRENT_JOBS=$((CURRENT_JOBS + 1))

# 2. SPD Run (Best Seed from CSV: 2)
wait_for_queue
run_finetune "TRUE" "FALSE" "FALSE" "SAC_WIPE_ICRA_SPD_LR1e-4_H150_EIGENCLAMP16" "2" &
CURRENT_JOBS=$((CURRENT_JOBS + 1))

wait

echo ""
echo "🎉  Domain Randomization Fine-Tuning complete!"
