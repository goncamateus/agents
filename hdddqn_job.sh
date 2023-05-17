#!/bin/bash
#SBATCH --job-name=hddqn_agent
#SBATCH --ntasks=1
#SBATCH --mem 24G
#SBATCH -c 32
#SBATCH -o hddqn_agent.log
#SBATCH --gpus=1

# Load modules and activate python environment
module use /opt/easybuild/modules/all/
module load Python3.10 Xvfb freeglut glew MuJoCo
source $HOME/.pyvenvs/rl3090/bin/activate


# Run the script
python run_hdddqn.py --cuda --gym-id $1 --total-timesteps 1000000 --capture-video --track
