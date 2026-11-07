
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

print("Creating Lunar Lander environment...")

# Create environment
env = gym.make("LunarLander-v3")

# Monitor training performance
env = Monitor(env)

print("Environment created successfully!")

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

print("Starting AI training...")

# Train the AI
model.learn(total_timesteps=500000)

# Save trained model
model.save("models/rocket_ppo")

print("Training completed!")
print("Model saved successfully!")

env.close()
