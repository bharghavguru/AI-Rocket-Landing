
import gymnasium as gym
from stable_baselines3 import PPO
from gymnasium.wrappers import RecordVideo

# Create video folder automatically
env = gym.make(
    "LunarLander-v3",
    render_mode="rgb_array"
)

# Record the first 5 episodes
env = RecordVideo(
    env,
    video_folder="videos",
    episode_trigger=lambda episode_id: episode_id < 5
)

# Load trained model
model = PPO.load("models/rocket_ppo")

for episode in range(5):

    observation, info = env.reset()
    done = False

    while not done:

        action, _ = model.predict(
            observation,
            deterministic=True
        )

        observation, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

env.close()

print("Video recording completed!")
