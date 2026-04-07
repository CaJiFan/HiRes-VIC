#!/bin/bash
#
# --- Slurm Configuration ---
#SBATCH --job-name=Tilted_Sweep
#SBATCH --output=logs/tilted_sweep_%j.txt
#SBATCH --error=logs/tilted_sweep_%j.err   
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64                    # 64 CPUs total (8 concurrent runs * 8 cores = 64)
#SBATCH --mem=150G                            # Plenty of RAM for 8 concurrent runs
#SBATCH --time=48:00:00                       
#SBATCH --gres=gpu:lovelace:1                 # Request exactly ONE GPU

echo "=========================================================="
echo "Starting Packed Slurm Job with Rolling Queue (Max 8)"
echo "Running on node: ${HOSTNAME}"
echo "Allocated GPU: ${CUDA_VISIBLE_DEVICES}"
echo "=========================================================="

# Safety check for logs directory
mkdir -p logs

eval "$(conda shell.bash hook)"
conda activate tfm

# --- Admin Required L40S Fixes ---
export NCCL_P2P_DISABLE=1
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024
export MALLOC_ARENA_MAX=1

# --- PYTORCH MIG BUG FIX ---
export PYTORCH_NVML_BASED_CUDA_CHECK=1

run_train() {
    ENV_NAME=$1
    ALGO=$2
    STEPS=$3
    USE_SPD=$4
    USE_LG=$5
    NUM_MARKERS=$6
    STIFF_PENALTY=$7
    EXP_NAME=$8
    SEED=$9

    echo "=================================================="
    echo "Starting $ALGO on $ENV_NAME for $STEPS steps..."
    echo "Date: $(date)"
    echo "=================================================="

    # 1. Start with the base arguments
    ARGS=(
        scripts/train.py
        --env "$ENV_NAME"
        --algorithm "$ALGO"
        --total_timesteps "$STEPS"
        --run_name "$EXP_NAME"
        --num_markers "$NUM_MARKERS"
        --stiff_penalty "$STIFF_PENALTY"
        --kp_min "$KP_MIN"
        --kp_max "$KP_MAX"
        --seed "$SEED"
    )

    # 2. Add conditional flags
    if [ "$USE_SPD" == "TRUE" ]; then
        ARGS+=(--use_spd)
    fi

    if [ "$USE_LG" == "TRUE" ]; then
        ARGS+=(--use_lie)
    fi

    # 3. Execute the command
    python3 "${ARGS[@]}" > "logs/${ENV_NAME}_${ALGO}_${EXP_NAME}_SEED_${SEED}.log" 2>&1
    
    echo "Finished $ALGO on $EXP_NAME"
}

# --- A helper function to enforce the 8-job limit ---
wait_for_queue() {
    # Check how many background processes this script currently owns
    while [ $(jobs -p | wc -l) -ge 8 ]; do
        sleep 30  # Wait 30 seconds before checking again
    done
}

# --- Define the Shared Parameters ---
TASK="TiltedWipe"
ALGO="SAC"
STIFF_PENALTY=0.01
KP_MIN=1
KP_MAX=200

for NUM_MARKERS in 5
do
    for SEED in 3 2 1 0 
    do
        run_train $TASK $ALGO 3_500_000 "TRUE" "TRUE" $NUM_MARKERS $STIFF_PENALTY "FULL_GRL_C0_NM${NUM_MARKERS}_SP${STIFF_PENALTY}_KP${KP_MIN}_${KP_MAX}" $SEED &
        wait_for_queue

        run_train $TASK $ALGO 3_500_000 "TRUE" "FALSE" $NUM_MARKERS $STIFF_PENALTY "SPD_ONLY_C0_NM${NUM_MARKERS}_SP${STIFF_PENALTY}_KP${KP_MIN}_${KP_MAX}" $SEED &
        wait_for_queue

        run_train $TASK $ALGO 3_500_000 "FALSE" "TRUE" $NUM_MARKERS $STIFF_PENALTY "LIE_ONLY_C0_NM${NUM_MARKERS}_SP${STIFF_PENALTY}_KP${KP_MIN}_${KP_MAX}" $SEED &
        wait_for_queue

        run_train $TASK $ALGO 3_500_000 "FALSE" "FALSE" $NUM_MARKERS $STIFF_PENALTY "BASELINE_C0_NM${NUM_MARKERS}_SP${STIFF_PENALTY}_KP${KP_MIN}_${KP_MAX}" $SEED &
        wait_for_queue
    done
done

# After all 16 jobs have been pushed through the queue, wait for the final batch of up to 8 to finish.
echo "All 16 experiments queued. Waiting for the final batch to finish..."
wait

echo "✅ All Door experiments completed successfully!"

conda deactivate