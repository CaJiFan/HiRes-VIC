#!/bin/bash


# WIPE environment
python3 scripts/train.py --env Wipe --algorithm PPO --total_timesteps 1_000_000 
python3 scripts/train.py --env Wipe --algorithm SAC --total_timesteps 1_000_000 



# NutAssemblyRound environment
python3 scripts/train.py --env NutAssemblyRound --algorithm PPO --total_timesteps 1_000_000 
python3 scripts/train.py --env NutAssemblyRound --algorithm SAC --total_timesteps 1_000_000


