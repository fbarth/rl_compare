"""
Module to train a PPO model with a MLP policy on the CarRacing-v1 environment with discrete actions.
"""

import sys
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env

BASE_PATH = "car_racing_discreto_MLP_ppo_env8"
RESULTS_PATH = f"./results/{BASE_PATH}/"
MODEL_PATH = f"./models/{BASE_PATH}"
ENV_NAME = "CarRacing-v2"

def train():
    new_logger = configure(RESULTS_PATH, ["stdout", "csv", "tensorboard"])

    env = gym.make(
        ENV_NAME,
        render_mode="rgb_array",
        lap_complete_percent=0.95,
        domain_randomize=False,
        continuous=False
    )
    vec_env = make_vec_env(
        ENV_NAME,
        n_envs=8,
        env_kwargs={
            "render_mode": "rgb_array",
            "lap_complete_percent": 0.95,
            "domain_randomize": False,
            "continuous": False
        }
    )

    # Stable-baseline PPO usage docs:
    # https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=1e-4,
        n_steps=512,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        vf_coef=0.5,
        ent_coef=0.0,
        max_grad_norm=0.5,
        tensorboard_log=None,
        verbose=1
    )

    model.set_logger(new_logger)
    model.learn(total_timesteps=1_000_000)
    model.save(MODEL_PATH)

    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    print(f'Mean reward: {mean_reward} +/- {std_reward:.2f}')
    env.close()
    return 0

def test():
    model = PPO.load(MODEL_PATH)

    print("Trained model. Testing...")
    env = gym.make(
        ENV_NAME,
        render_mode="human",
        lap_complete_percent=0.95,
        domain_randomize=False,
        continuous=False
    )

    (obs,_) = env.reset()
    for i in range(1000):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()
    env.close()
    return 0

def main():
    train_model = input("Train model? (Y/n): ")
    if train_model.lower() == "n":
        sys.exit(test())
    train()
    sys.exit(test())

if __name__ == "__main__":
    main()
