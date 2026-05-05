#!/bin/bash
#
# --- Slurm Configuration ---
#SBATCH --job-name=Tilted_ShortHorizon
#SBATCH --output=logs/tilted_short_horizon_%j.txt
#SBATCH --error=logs/tilted_short_horizon_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48                    # 48 CPUs total (6 concurrent runs * 8 cores = 48)
#SBATCH --mem=150G                            # Plenty of RAM for 6 concurrent runs
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

run_train() {
    ENV_NAME=$1
    USE_SPD=$2
    USE_LG=$3
    NUM_MARKERS=$4
    EXP_NAME=$5
    SEED=$6

    echo "=================================================="
    echo "Starting SAC on $ENV_NAME for $STEPS steps..."
    echo "Date: $(date)"
    echo "=================================================="

    # 1. Start with the base arguments
    ARGS=(
        src/train_fixed.py
        --env "$ENV_NAME"
        --run_name "$EXP_NAME"
        --num_markers "$NUM_MARKERS"
        --fixed_kp "$FIXED_KP"
        --seed "$SEED"
    )

    # 2. Add conditional flags
    if [ "$USE_SPD" == "TRUE" ]; then
        ARGS+=(--use_spd)
    fi

    if [ "$USE_SPD" == "FIXED" ]; then
        ARGS+=(--use_fixed)
    fi

    if [ "$USE_LG" == "TRUE" ]; then
        ARGS+=(--use_lie)
    fi

    # 3. Execute the command
    python3 "${ARGS[@]}" > "logs/${ENV_NAME}_SAC_${EXP_NAME}_SEED_${SEED}.log" 2>&1
    
    echo "Finished SAC on $EXP_NAME"
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
NUM_MARKERS=5
GAMMA=0.90

for FIXED_KP in 150 300 500
do
    for SEED in 1 0 
    do    
        run_train $TASK "FIXED" "" $NUM_MARKERS "FOS_FIXED_C1_NM${NUM_MARKERS}_KP${FIXED_KP}_GAMMA${GAMMA}" $SEED &
        wait_for_queue

        # run_train $TASK "TRUE" "FALSE" $NUM_MARKERS "FOS_SPD_ONLY_C1_NM${NUM_MARKERS}_KP1_300" $SEED &
        # wait_for_queue

        # run_train $TASK "FALSE" "TRUE" $NUM_MARKERS "FOS_LIE_ONLY_C1_NM${NUM_MARKERS}_KP1_300" $SEED &
        # wait_for_queue

        # run_train $TASK "FALSE" "FALSE" $NUM_MARKERS "FOS_BASELINE_C1_NM${NUM_MARKERS}_KP1_300" $SEED &
        # wait_for_queue
    done
done

# After all 16 jobs have been pushed through the queue, wait for the final batch of up to 8 to finish.
echo "All experiments queued. Waiting for the final batch to finish..."
wait

echo "✅ All Door experiments completed successfully!"

conda deactivate