#!/bin/bash

# Carpole environment
cd src
nohup python cart_pole_dqn.py &
nohup python cart_pole_ppo_env_1.py &
nohup python cart_pole_ppo_env_4.py &
nohup python cart_pole_ppo_env_8.py &
nohup python cartpole_a2c_env1.py & 

