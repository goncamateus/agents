#!/bin/bash
#SBATCH --job-name=rainbow_agent
#SBATCH --ntasks=1
#SBATCH --mem 24G
#SBATCH -c 32
#SBATCH -o rainbow_agent.log
#SBATCH --gpus=1
#SBATCH -p short
#SBATCH --mail-type=FAIL,END,ARRAY_TASKS
#SBATCH --mail-user=mgm4@cin.ufpe.br

# Load modules and activate python environment
module load Python3.10 Xvfb freeglut glew MuJoCo
source $HOME/.pyvenvs/rl/bin/activate


# Run the script
python run_rainbow.py --cuda --gym-id $1 --total-timesteps 1000000 --capture-video --track
