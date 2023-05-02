#!/bin/bash
#SBATCH --job-name=ppo_strat_agent
#SBATCH --ntasks=1
#SBATCH --mem 64G
#SBATCH -c 32
#SBATCH -o ppo_strat.log
#SBATCH --gpus=1

# Load modules and activate python environment
module load Python3.10 Xvfb freeglut
source $HOME/.pyenvs/rl/bin/activate

# Run the script
python run_ppo.py -cuda --gym-id $1 --total-timesteps 1000000 \
    --capture-video --num-envs 16 --track --video-freq 10
