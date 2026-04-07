#!/bin/bash
#
# --- Slurm Configuration ---
#SBATCH --job-name=Tilted_Sweep
#SBATCH --output=logs/tilted_sweep_%j.txt
#SBATCH --error=logs/tilted_sweep_%j.err   
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64                    # 64 CPUs total (16 per experiment)
#SBATCH --mem=150G                            # ~35GB per experiment
#SBATCH --time=48:00:00                       
#SBATCH --gres=gpu:lovelace:1                 # Request exactly ONE GPU

echo "=========================================================="
echo "Starting Packed Slurm Job"
echo "Running on node: ${HOSTNAME}"
echo "Allocated GPU: ${CUDA_VISIBLE_DEVICES}"
echo "=========================================================="

eval "$(conda shell.bash hook)"
conda activate tfm

# --- Admin Required L40S Fixes ---
export NCCL_P2P_DISABLE=1
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024
export MALLOC_ARENA_MAX=1

# --- Define the Shared Parameters ---
ENV_NAME="TiltedWipe"
ALGO="SAC"
SEED=2
# Make sure your train.py is hardcoded to n_envs=16, or pass it here if your script allows it!

echo "🚀 Launching 4 parallel experiments on GPU ${CUDA_VISIBLE_DEVICES}..."

# --- Launch All 4 Combinations Simultaneously ---
# The '&' at the end of each line pushes the process to the background
# so the script moves immediately to the next line.

# 1. Baseline (No SPD, No Lie Group)
python src/train.py --env $ENV_NAME --algorithm $ALGO --run_name BASELINE --seed $SEED &

# 2. SPD Only
python src/train.py --env $ENV_NAME --algorithm $ALGO --run_name SPD_ONLY --use_spd --seed $SEED &

# 3. Lie Group Only
python src/train.py --env $ENV_NAME --algorithm $ALGO --run_name LG_ONLY --use_lie --seed $SEED &

# 4. SPD + Lie Group (Full Geometric)
python src/train.py --env $ENV_NAME --algorithm $ALGO --run_name FULL_GRL --use_spd --use_lie --seed $SEED &

# --- WAIT COMMAND ---
# This is CRITICAL. 'wait' tells Slurm not to kill the job until 
# all 4 of the background processes finish.
wait

echo "✅ All 4 experiments completed successfully!"
conda deactivate