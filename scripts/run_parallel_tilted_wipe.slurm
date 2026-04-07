#!/bin/bash
#
# --- Slurm Configuration ---
#SBATCH --job-name=RoboWipe_Array
#SBATCH --array=0-31                          # 4 Configs * 2 Markers * 4 Seeds = 32 jobs
#SBATCH --output=logs/wipe_%A_%a.out          # %A is the array master ID, %a is the specific task ID
#SBATCH --error=logs/wipe_%A_%a.err   
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8                     # EXACTLY 8 CORES for n_envs=8!
#SBATCH --mem=160G                             # RAM for 8 envs + Replay Buffer
#SBATCH --time=48:00:00                       
#SBATCH --gres=gpu:lovelace:1                 # 1 GPU per job (Slurm will run 4 concurrently!)

# Safety check for logs
mkdir -p logs

# Activate your environment (adjust as needed)
eval "$(conda shell.bash hook)"
conda activate tfm

# --- Global Constants ---
TASK="TiltedWipe"
STIFF_PENALTY=0.01
KP_MIN=1
KP_MAX=1000

# --- Define the Hyperparameter Grid ---
# CONFIGS=("FULL_GRL" "SPD_ONLY" "LIE_ONLY" "BASELINE")
CONFIGS=("BASELINE")
MARKERS=(10 5)
SEEDS=(3 2 1 0)

# --- Decode the Slurm Array ID ($SLURM_ARRAY_TASK_ID) ---
# This calculates which specific combination this exact job should run
NUM_SEEDS=${#SEEDS[@]}
NUM_MARKERS=${#MARKERS[@]}

SEED_IDX=$(($SLURM_ARRAY_TASK_ID % $NUM_SEEDS))
TEMP=$(($SLURM_ARRAY_TASK_ID / $NUM_SEEDS))
MARKER_IDX=$(($TEMP % $NUM_MARKERS))
CONFIG_IDX=$(($TEMP / $NUM_MARKERS))

CURRENT_CONFIG=${CONFIGS[$CONFIG_IDX]}
CURRENT_MARKER=${MARKERS[$MARKER_IDX]}
CURRENT_SEED=${SEEDS[$SEED_IDX]}
EXP_NAME="${CURRENT_CONFIG}_C0_NM${CURRENT_MARKER}_SP${STIFF_PENALTY}_KP${KP_MIN}_${KP_MAX}"

echo "=========================================================="
echo "Job Array ID: ${SLURM_ARRAY_TASK_ID}"
echo "Configuration: ${CURRENT_CONFIG}"
echo "Markers: ${CURRENT_MARKER}"
echo "Seed: ${CURRENT_SEED}"
echo "Running on node: ${HOSTNAME}"
echo "Allocated GPU: ${CUDA_VISIBLE_DEVICES}"
echo "=========================================================="

# --- 1. Base Arguments ---
ARGS=(
    scripts/train.py
    --env "$TASK"
    --algorithm "SAC"
    --total_timesteps 3500000
    --run_name "$EXP_NAME"
    --num_markers "$CURRENT_MARKER"
    --stiff_penalty "$STIFF_PENALTY"
    --kp_min "$KP_MIN"
    --kp_max "$KP_MAX"
    --seed "$CURRENT_SEED"
)

# --- 2. Add Conditional Flags based on CURRENT_CONFIG ---
if [ "$CURRENT_CONFIG" == "FULL_GRL" ]; then
    ARGS+=(--use_spd --use_lie)
elif [ "$CURRENT_CONFIG" == "SPD_ONLY" ]; then
    ARGS+=(--use_spd)
elif [ "$CURRENT_CONFIG" == "LIE_ONLY" ]; then
    ARGS+=(--use_lie)
fi
# BASELINE gets no extra flags.

# --- 3. Execute ---
python3 "${ARGS[@]}"

echo "✅ Finished training ${EXP_NAME} (Seed: ${CURRENT_SEED})"