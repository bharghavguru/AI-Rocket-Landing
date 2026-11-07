
import os
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from custom_env import CustomRewardWrapper


# Create models directory
os.makedirs("models", exist_ok=True)

# Create LunarLander environment
env = gym.make("LunarLander-v3")

# Apply custom reward wrapper
env = CustomRewardWrapper(env)

# Monitor the environment
env = Monitor(env)

# Create PPO model
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=0.0003,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    device="cpu"
)

# Train the model
print("Training PPO with Custom Reward Shaping...")

model.learn(total_timesteps=500000)

# Save the custom model
model.save("models/rocket_ppo_custom")

print("Custom reward training completed!")
print("Model saved successfully.")

env.close()
