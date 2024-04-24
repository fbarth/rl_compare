#!/bin/bash

python run_model.py CartPole-v1 models/cartpole_dqn DQN
python run_model.py CartPole-v1 models/cartpole_ppo_env_1 PPO
python run_model.py BipedalWalker-v3 models/bipedal_walker_a2c A2C
