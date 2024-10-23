# Implementations of Reinforcement Learning methods by goncamateus


## Table of Contents

- [Implementations of Reinforcement Learning methods by goncamateus](#implementations-of-reinforcement-learning-methods-by-goncamateus)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Project Structure](#project-structure)

## Introduction

This repository contains implementations of Reinforcement Learning methods. The implementations are based on CleanRL agents. The goal is to guide myself and others in the learning process of Reinforcement Learning.


## Installation
```bash
poetry install
```

## Usage

```bash
poetry run python train.py
```

## Project Structure

- agents/
    - networks/
        - network.py (abstract)
        - value_function/
            - mlp.py
            - cnn.py
        - policy/
            - mlp.py
            - cnn.py
            - gaussian.py
    - methods/
        - method.py (abstract)
        - dqn.py
        - ddpg.py
        - ppo.py
        - sac.py
    - experience/
      - base_buffer.py (abstract)
      - replay_buffer.py
      - prioritized_replay_buffer
          - segment_tree.py
    - utils/
        - experiment
          - rollout.py
          - hyperparameter_manager.py
          - logger.py
          - maker.py
        - noise/
          - ornstein_uhlenbeck.py
          - gaussian.py
- train.py