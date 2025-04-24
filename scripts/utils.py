import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from env.pen_spin_env import PenSpinEnv

def train_agent(total_timesteps=100000, save_path="models/ppo_pen_spin_success.zip"):
    env = PenSpinEnv(render=False)

    # PPO agent with default MLP policy
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./results/training_logs"
    )

    # Optional: Save checkpoints every 10k steps
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path="./models", name_prefix="ppo_checkpoint")

    # Train the model
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)

    # Save final model
    model.save(save_path)
    print(f"\n✅ Training complete. Model saved to: {save_path}")

    env.close()

if __name__ == "__main__":
    train_agent()
