# $1 eh o caminho para a pasta do rSoccer
# $2 eh o caminho para a pasta do agent
sudo docker run -it --network host -e DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix -v $1:/rl/rSoccer -v $2:/rl/agents --rm --runtime=nvidia --gpus all deep_chair /bin/bash