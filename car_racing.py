
from stable_baselines3 import A2C
import gymnasium as gym
from stable_baselines3.common.logger import configure
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env 
from stable_baselines3.common.vec_env import VecFrameStack

tmp_path = "./results/car_racing_discrete_a2c_cnn/"
new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])

env = gym.make("CarRacing-v2")
vec_env = make_vec_env("CarRacing-v2", n_envs=8, env_kwargs={"domain_randomize": False, "continuous": False})

n_stack = 12

env = VecFrameStack(vec_env, n_stack)
model = A2C("CnnPolicy", env=env)

model.set_logger(new_logger)
model.learn(total_timesteps=10_000)
model.save("models/car_racing_discrete_frameStack_a2c_cnn")

mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print(f'Mean reward: {mean_reward} +/- {std_reward:.2f}')

del model
model = A2C.load("models/car_racing_discrete_a2c_cnn")

print('modelo treinado')
env = gym.make("CarRacing-v2", render_mode='human')
(obs,_) = env.reset()
for i in range(1000):
    action, _state = model.predict(obs)
    obs, reward, done, truncated, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()