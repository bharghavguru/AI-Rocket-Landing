
import os
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from custom_env_v2 import CustomRewardV2


# Create models directory
os.makedirs("models", exist_ok=True)

# Create LunarLander environment
env = gym.make("LunarLander-v3")

# Apply Custom Reward V2
env = CustomRewardV2(env)

# Monitor environment
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
    device="cpu",
    seed=42
)

print("Training PPO with Custom Reward V2...")

# Train model
model.learn(total_timesteps=500000)

# Save separately
model.save("models/rocket_ppo_custom_v2")

print("Custom Reward V2 training completed!")
print("Model saved successfully.")

env.close()
