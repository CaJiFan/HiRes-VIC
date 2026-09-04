#!/bin/bash
#
# --- Slurm Configuration ---
#SBATCH --job-name=wipe_dr
#SBATCH --output=logs/wipe_dr_%j.txt
#SBATCH --error=logs/wipe_dr_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
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

# ── Fixed hyperparameters ──────────────────────────────────────────────────
TASK="TiltedWipe"
STEPS=1000000
HORIZON=150
N_ENVS=16
GAMMA=0.95
NUM_MARKERS=5
SEEDS=(0 1 2 3 4)
LR=3e-4
BATCH_SIZE=1024
GAMMA_START=0.95

# ── LLM Prior Setup ───────────────────────────────────────────────────────
LLM_W_INIT=0.40
LLM_W_FLOOR=0.05
LLM_ANNEAL_SCHEDULE="cosine"
LLM_ANNEAL=0
LLM_BACKEND=""
LLM_MODEL="llama3.2"
LLM_QUERY_INTERVAL=35

wait_for_queue() {
    while [ $(jobs -pr | wc -l) -ge 6 ]; do
        sleep 10
    done
}

# ── Helper functions ───────────────────────────────────────────────────────
run_geo_dr() {
    local USE_SPD=$1   # "TRUE" | "FALSE"
    local USE_LIE=$2   # "TRUE" | "FALSE"
    local USE_DIAG=$3  # "TRUE" | "FALSE"
    local EXP_NAME=$4
    local SEED=$5
    local LOAD_MODEL=$6 # Optional path to checkpoint to resume from

    echo "────────────────────────────────────────────────────────"
    echo "▶ GEO-DR  ${EXP_NAME} | SEED=${SEED}"
    echo "  SPD=${USE_SPD} LIE=${USE_LIE} DIAG=${USE_DIAG}"
    [ -n "$LOAD_MODEL" ] && echo "  Loading checkpoint: ${LOAD_MODEL}"
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
        --gamma_start "$GAMMA_START"
        --num_markers "$NUM_MARKERS"
        --seed "$SEED"
        --lr "$LR"
        --batch_size "$BATCH_SIZE"
        --record_video
        --camera_names "frontview"
        --primitive_init "teleport"
        --use_quality_reward
        --use_sequential_waypoints
        --use_domain_rand
    )
    [ "$USE_SPD"  == "TRUE" ] && ARGS+=(--use_spd)
    [ "$USE_LIE"  == "TRUE" ] && ARGS+=(--use_lie)
    [ "$USE_DIAG" == "TRUE" ] && ARGS+=(--use_diag)
    [ -n "$LOAD_MODEL" ] && [ -f "$LOAD_MODEL" ] && ARGS+=(--load_model "$LOAD_MODEL")

    LOG="logs/wipe/${TASK}_${EXP_NAME}_SEED_${SEED}.log"
    echo "  → ${LOG}"
    "${ARGS[@]}" > "$LOG" 2>&1
    echo "  ✅ Done: ${EXP_NAME} SEED=${SEED}"
}

run_llm_dr() {
    local USE_SPD=$1
    local USE_LIE=$2
    local USE_DIAG=$3
    local EXP_NAME=$4
    local SEED=$5
    local LOAD_MODEL=$6

    echo "────────────────────────────────────────────────────────"
    echo "▶ LLM-DR  ${EXP_NAME} | SEED=${SEED}"
    echo "  SPD=${USE_SPD} LIE=${USE_LIE} DIAG=${USE_DIAG}"
    [ -n "$LOAD_MODEL" ] && echo "  Loading checkpoint: ${LOAD_MODEL}"
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
        --primitive_init "teleport"
        --use_quality_reward
        --use_sequential_waypoints
        --use_domain_rand
        --use_llm_prior
        --llm_backend "$LLM_BACKEND"
        --llm_model "$LLM_MODEL"
        --llm_prior_weight "$LLM_W_INIT"
        --llm_anneal_steps "$LLM_ANNEAL"
        --llm_anneal_floor "$LLM_W_FLOOR"
        --llm_anneal_schedule "$LLM_ANNEAL_SCHEDULE"
        --llm_query_interval "$LLM_QUERY_INTERVAL"
    )
    [ "$USE_SPD"  == "TRUE" ] && ARGS+=(--use_spd)
    [ "$USE_LIE"  == "TRUE" ] && ARGS+=(--use_lie)
    [ "$USE_DIAG" == "TRUE" ] && ARGS+=(--use_diag)
    [ -n "$LOAD_MODEL" ] && [ -f "$LOAD_MODEL" ] && ARGS+=(--load_model "$LOAD_MODEL")

    LOG="logs/wipe/${TASK}_${EXP_NAME}_SEED_${SEED}.log"
    echo "  → ${LOG}"
    "${ARGS[@]}" > "$LOG" 2>&1
    echo "  ✅ Done: ${EXP_NAME} SEED=${SEED}"
}

TRAIN_PIDS=()

for SEED in "${SEEDS[@]}"; do
    echo "════════════════  SEED ${SEED}  ════════════════"

    # Pre-trained checkpoint paths from the NORMv1 runs
    BASE_CKPT="logs/best_models/SAC_WIPE_ICRA_BASELINE_LR3e-4_H150_HQSEQ_NM5_G0.95_NORMv1_SEED_${SEED}/best_success_model.zip"
    SPD_CKPT="logs/best_models/SAC_WIPE_ICRA_SPD_LR3e-4_H150_HQSEQ_NM5_G0.95_NORMv1_SEED_${SEED}/best_success_model.zip"

    # 1. BASELINE Fine-Tuning with Domain Randomization (Diagonal VIC)
    wait_for_queue
    run_geo_dr "FALSE" "FALSE" "FALSE" "WIPE_ICRA_BASELINE_DR_LR${LR}_H${HORIZON}_HQSEQ_G${GAMMA}" "$SEED" "$BASE_CKPT" &
    TRAIN_PIDS+=($!)

    # # 2. SPD MANIFOLD Fine-Tuning with Domain Randomization (Full 3x3 stiffness coupling)
    # wait_for_queue
    # run_geo_dr "TRUE"  "FALSE" "FALSE" "WIPE_ICRA_SPD_DR_LR${LR}_H${HORIZON}_HQSEQ_G${GAMMA}" "$SEED" "$SPD_CKPT" &
    # TRAIN_PIDS+=($!)

    # 3. SPD + LLM Fine-Tuning with Domain Randomization (Riemannian SPD + LLM semantic prior)
    # wait_for_queue
    # run_llm_dr "TRUE"  "FALSE" "FALSE" "WIPE_ICRA_SPDLLM_DR_LR${LR}_H${HORIZON}_HQSEQ_G${GAMMA}" "$SEED" "$SPD_CKPT" &
    # TRAIN_PIDS+=($!)
done

for SEED in "${SEEDS[@]}"; do
    echo "════════════════  SEED ${SEED}  ════════════════"

    # Pre-trained checkpoint paths from the NORMv1 runs
    BASE_CKPT="logs/best_models/SAC_WIPE_ICRA_BASELINE_LR3e-4_H150_HQSEQ_NM5_G0.95_NORMv1_SEED_${SEED}/best_success_model.zip"
    SPD_CKPT="logs/best_models/SAC_WIPE_ICRA_SPD_LR3e-4_H150_HQSEQ_NM5_G0.95_NORMv1_SEED_${SEED}/best_success_model.zip"

    # # 1. BASELINE Fine-Tuning with Domain Randomization (Diagonal VIC)
    # wait_for_queue
    # run_geo_dr "FALSE" "FALSE" "FALSE" "WIPE_ICRA_BASELINE_DR_LR${LR}_H${HORIZON}_HQSEQ_G${GAMMA}" "$SEED" "$BASE_CKPT" &
    # TRAIN_PIDS+=($!)

    # 2. SPD MANIFOLD Fine-Tuning with Domain Randomization (Full 3x3 stiffness coupling)
    wait_for_queue
    run_geo_dr "TRUE"  "FALSE" "FALSE" "WIPE_ICRA_SPD_DR_LR${LR}_H${HORIZON}_HQSEQ_G${GAMMA}" "$SEED" "$SPD_CKPT" &
    TRAIN_PIDS+=($!)

    # 3. SPD + LLM Fine-Tuning with Domain Randomization (Riemannian SPD + LLM semantic prior)
    # wait_for_queue
    # run_llm_dr "TRUE"  "FALSE" "FALSE" "WIPE_ICRA_SPDLLM_DR_LR${LR}_H${HORIZON}_HQSEQ_G${GAMMA}" "$SEED" "$SPD_CKPT" &
    # TRAIN_PIDS+=($!)
done

# Wait for all background training runs to complete
wait "${TRAIN_PIDS[@]}"

echo ""
echo "🎉 All TiltedWipe Domain Randomization experiments complete!"

conda deactivate
