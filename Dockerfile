FROM nvidia/cuda:12.2.0-base-ubuntu20.04
# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
# Install system dependencies
RUN apt update && apt upgrade -y
RUN apt update && apt install -y build-essential \
    cmake \
    libopenmpi-dev \
    zlib1g-dev \
    libgl1-mesa-glx \
    libglfw3 \
    libglfw3-dev \
    git \
    libncurses5-dev \
    libgdbm-dev \
    libnss3-dev \
    libssl-dev \
    libreadline-dev \
    libffi-dev \
    libsqlite3-dev \
    wget \
    libbz2-dev \
    xvfb \
    freeglut3-dev \
    freeglut3 \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    libxext-dev \
    libxt-dev

RUN wget https://www.python.org/ftp/python/3.10.12/Python-3.10.12.tgz
RUN tar -xf /Python-3.10.12.tgz
WORKDIR /Python-3.10.12
RUN ./configure --enable-optimizations
RUN make -j $(nproc)
RUN make altinstall

RUN python3.10 -m venv /deep
ENV PATH="/deep/bin:$PATH"

WORKDIR /rl
# # Install python dependencies
RUN git clone https://github.com/goncamateus/agents.git
RUN git clone https://github.com/robocin/rSoccer.git
RUN ["/bin/bash", "-c", "source /deep/bin/activate"]
COPY requirements.txt .
RUN pip install -r requirements.txt
WORKDIR /rl/rSoccer
RUN git checkout cadeira_deep
RUN pip install -e .
WORKDIR /rl/agents
RUN git checkout cadeira_deep
RUN git config --global --add safe.directory /rl/agents
COPY wandb_hack.py /deep/lib/python3.10/site-packages/wandb/integration/gym/__init__.py
CMD [ "python", "run_sac.py", "--gym-id", "SSLPathPlanning-v0", "--total-timesteps", "150000", "--capture-video", "--num-envs", "16", "--track" ]