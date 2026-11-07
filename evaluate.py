
import gymnasium as gym
from stable_baselines3 import PPO

# Create the environment
env = gym.make(
    "LunarLander-v3",
    render_mode="human"
)

# Load trained AI
model = PPO.load("models/rocket_ppo")

print("Trained AI loaded successfully!")

# Evaluate over 10 episodes
total_rewards = []

for episode in range(10):

    observation, info = env.reset()
    done = False
    episode_reward = 0

    while not done:

        # AI chooses an action
        action, _states = model.predict(
            observation,
            deterministic=True
        )

        # Perform action
        observation, reward, terminated, truncated, info = env.step(action)

        episode_reward += reward

        done = terminated or truncated

    total_rewards.append(episode_reward)

    print(
        f"Episode {episode + 1}: "
        f"Reward = {episode_reward:.2f}"
    )

# Calculate average reward
average_reward = sum(total_rewards) / len(total_rewards)

print("--------------------------------")
print("Average reward:", average_reward)
print("--------------------------------")

env.close()
