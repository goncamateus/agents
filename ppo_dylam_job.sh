#!/bin/bash
#SBATCH --job-name=ppo_dylam_agent
#SBATCH --ntasks=1
#SBATCH --mem 24G
#SBATCH -c 32
#SBATCH -o ppo_dylam.log
#SBATCH --gpus=1

# Load modules and activate python environment
module load Python3.10 Xvfb freeglut glew MuJoCo
source $HOME/.pyvenvs/rl3090/bin/activate

# Run the script
cd $HOME/doc/rl/agents
python run_ppo_strat.py --cuda --gym-id $1 --total-timesteps 1000000 \
    --capture-video --num-envs 16 --track --video-freq 10  --dylam
