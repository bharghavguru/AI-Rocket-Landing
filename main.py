
import gymnasium as gym

# Create the Lunar Lander environment
env = gym.make("LunarLander-v3", render_mode="human")

# Start a new episode
observation, info = env.reset()

print("Starting Rocket Simulation!")

# Run 1000 random actions
for step in range(1000):

    # Choose a random action
    action = env.action_space.sample()

    # Perform the action
    observation, reward, terminated, truncated, info = env.step(action)

    # Print information
    print("Step:", step)
    print("Observation:", observation)
    print("Reward:", reward)

    # Check whether the episode ended
    if terminated or truncated:
        print("Episode finished!")

        observation, info = env.reset()

# Close the environment
env.close()

print("Simulation completed!")
