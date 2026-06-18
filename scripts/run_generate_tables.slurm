#!/bin/bash
#
# --- Slurm Configuration ---
#SBATCH --job-name=generate_tables_GRIP
#SBATCH --output=logs/generate_tables_GRIP_%j.txt
#SBATCH --error=logs/generate_tables_GRIP_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:lovelace:1


source ~/.bashrc
conda activate tfm

export PYTHONPATH=/home/cjimenez/projects/HiRes-VIC:$PYTHONPATH

export OLLAMA_MODELS="/home/cjimenez/ollama_models"
UNIQUE_PORT=$((11000 + (SLURM_JOB_ID % 1000)))
export OLLAMA_HOST="127.0.0.1:${UNIQUE_PORT}"

echo "Starting Ollama server on port ${UNIQUE_PORT}..."
~/.local/bin/ollama serve &
OLLAMA_PID=$!
sleep 10

python3 /home/cjimenez/projects/HiRes-VIC/scripts/generate_tables.py

echo "Stopping Ollama server..."
kill $OLLAMA_PID
