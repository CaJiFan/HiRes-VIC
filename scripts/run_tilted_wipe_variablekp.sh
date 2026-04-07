#!/bin/bash
#
# --- Slurm Configuration ---
#SBATCH --job-name=Fixed_Ablation
#SBATCH --output=logs/fixed_ablation_%j.txt
#SBATCH --error=logs/fixed_ablation_%j.err   
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64                    # 8 concurrent runs * 8 n_envs = 64 Cores!
#SBATCH --mem=150G                            # Plenty of RAM for 8 vectorized SAC runs
#SBATCH --time=48:00:00                       
#SBATCH --gres=gpu:lovelace:1                 # Request exactly ONE GPU

echo "=========================================================="
echo "Starting Fixed Impedance Ablation Sweep"
echo "Running on node: ${HOSTNAME}"
echo "Allocated GPU: ${CUDA_VISIBLE_DEVICES}"
echo "=========================================================="

# Safety check for logs directory
mkdir -p logs

eval "$(conda shell.bash hook)"
conda activate tfm

# --- STRICT DETERMINISM & CLUSTER FIXES ---
export NCCL_P2P_DISABLE=1
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024
export MALLOC_ARENA_MAX=1

# --- A helper function to enforce the 8-job GPU packing limit ---
wait_for_queue() {
    # If there are 8 or more background jobs running, pause and wait.
    while [ $(jobs -p | wc -l) -ge 6 ]; do
        sleep 30  
    done
}

# --- Sweep Parameters ---
ENV_NAME="TiltedWipe"
NUM_MARKERS=5
CONTROLLER_TYPE="variable_kp"  # This script is specifically for the variable stiffness controller ablation

# 2 Controllers x 4 Seeds = exactly 8 jobs.
SEEDS=(0 1 2 3)

# --- Launch the Rolling Queue ---
for SEED in "${SEEDS[@]}"; do
        
    # Build a clean Run Name for WandB and saving
    RUN_NAME="${CONTROLLER_TYPE^^}_C1_NM${NUM_MARKERS}"
    
    echo "Launching: $RUN_NAME"
    
    # Execute the Python script in the BACKGROUND (&)
    python3 src/train_fixed.py \
        --env "$ENV_NAME" \
        --num_markers "$NUM_MARKERS" \
        --controller_type "$CONTROLLER_TYPE" \
        --seed "$SEED" \
        --run_name "$RUN_NAME" > logs/TILTEDWIPE_SAC_${RUN_NAME}_${SEED}.txt 2>&1 &
        
    # Check the queue to ensure we don't exceed 8 concurrent processes
    wait_for_queue
        
done

# Wait for the final batch to finish
echo "All fixed ablation jobs queued. Waiting for completion..."
wait

echo "✅ Fixed Impedance Ablation Sweep Completed!"
conda deactivate