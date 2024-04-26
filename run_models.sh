#!/bin/bash

python run_model.py CartPole-v1 models/cartpole_dqn DQN
python run_model.py CartPole-v1 models/cartpole_ppo_env_1 PPO

python run_model.py BipedalWalker-v3 models/bipedal_walker_a2c A2C
python run_model.py BipedalWalker-v3 models/bipedalwalker_ppo PPO
python run_model.py BipedalWalker-v3 models/bipedalwalker_ppo_hugging_face PPO

python run_model.py CarRacing-v2 models/car_racing_discreto_cnn_ppo_env_8 PPO

python run_model.py LunarLander-v2 models/lunar_lander_ppo_env_1 PPO
