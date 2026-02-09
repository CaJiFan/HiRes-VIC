#!/bin/bash

# Door environment
python3 scripts/train.py --env Door --algorithm PPO --total_timesteps 500_000 
python3 scripts/train.py --env Door --algorithm SAC --total_timesteps 500_000 

# NutAssemblySquare environment
python3 scripts/train.py --env NutAssemblySquare --algorithm PPO --total_timesteps 500_000 
python3 scripts/train.py --env NutAssemblySquare --algorithm SAC --total_timesteps 500_000

# WIPE environment
python3 scripts/train.py --env Wipe --algorithm PPO --total_timesteps 1_000_000 
python3 scripts/train.py --env Wipe --algorithm SAC --total_timesteps 1_000_000 


# NutAssemblyRound environment
python3 scripts/train.py --env NutAssemblyRound --algorithm PPO --total_timesteps 1_000_000 
python3 scripts/train.py --env NutAssemblyRound --algorithm SAC --total_timesteps 1_000_000


