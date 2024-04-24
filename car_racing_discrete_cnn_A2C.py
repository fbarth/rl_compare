
from stable_baselines3 import A2C
import gymnasium as gym
from stable_baselines3.common.logger import configure
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env 
from stable_baselines3.common.vec_env import VecFrameStack
import torch
import gc
import cv2
import numpy as np
gc.collect()
torch.cuda.empty_cache()


def train():
  tmp_path = "./results/car_racing_discrete_a2c_cnn/"
  new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])

  env = gym.make("CarRacing-v2")
  vec_env = make_vec_env("CarRacing-v2", n_envs=8, env_kwargs={"domain_randomize": False, "continuous": False})

  n_stack = 4

  env = VecFrameStack(vec_env, n_stack)
  model = A2C("CnnPolicy", 
      env=env,
      ent_coef=0.0,
      gae_lambda=0.95,
      vf_coef=0.5,
      learning_rate=2e-4,
      gamma=0.99,
      tensorboard_log=None,
    )

  model.set_logger(new_logger)
  model.learn(total_timesteps=100_000)
  model.save("models/car_racing_discrete_frameStack_a2c_cnn")

  mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
  print(f'Mean reward: {mean_reward} +/- {std_reward:.2f}')

def preprocess_observation(observation):
    resized_observation = cv2.resize(observation, (96, 96))
    # grayscale_observation = cv2.cvtColor(resized_observation, cv2.COLOR_RGB2GRAY)
    stacked_observation = np.stack([resized_observation] * 12, axis=0)
    return stacked_observation

def test():
  model = A2C.load("models/car_racing_discrete_frameStack_a2c_cnn")

  print('modelo treinado')
  env = gym.make("CarRacing-v2", render_mode='human', continuous=False)
  (obs,_) = env.reset()
  for i in range(1000):
    preprocessed_obs = preprocess_observation(obs)
    action, _state = model.predict(preprocessed_obs)
    obs, reward, done, truncated, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()

if __name__ == "__main__":
    should_train = input("Train? (y/n): ")
    if should_train == "y":
        train()
    test()