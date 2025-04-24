import os
import gymnasium as gym
import torch
from stable_baselines3 import PPO
from env.pen_spin_env import PenSpinEnv

def evaluate_agent(model_path, num_episodes=5, render=True):
    env = PenSpinEnv(render=render)

    model = PPO.load(model_path, env=env)

    print(f"\nEvaluating PPO model: {model_path}\n")
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        step = 0
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            step += 1
            if render:
                env.render()
        print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f} | Steps = {step}")
    env.close()

if __name__ == "__main__":
    model_file = os.path.join("models", "ppo_pen_spin_success.zip")
    evaluate_agent(model_path=model_file)
