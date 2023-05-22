#!/bin/bash
#SBATCH --job-name=sac_strat_agent
#SBATCH --ntasks=1
#SBATCH --mem 24G
#SBATCH -c 32
#SBATCH -o sac_strat_agent.log
#SBATCH --gpus=1

# Load modules and activate python environment
module load Python3.10 Xvfb freeglut glew MuJoCo
source $HOME/.pyvenvs/rl3090/bin/activate


# Run the script
python run_sac_strat.py --cuda --gym-id $1 --total-timesteps 1000000\
    --capture-video --num-envs 16 --track
