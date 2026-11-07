
import gymnasium as gym
from stable_baselines3 import PPO


EPISODES = 20

# Load both trained models
original_model = PPO.load("models/rocket_ppo")
custom_model = PPO.load("models/rocket_ppo_custom")


def evaluate_model(model, model_name):
    env = gym.make("LunarLander-v3")

    episode_rewards = []

    print(f"\nEvaluating {model_name}...")

    for episode in range(EPISODES):

        observation, info = env.reset(seed=episode)

        total_reward = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):

            action, _ = model.predict(
                observation,
                deterministic=True
            )

            observation, reward, terminated, truncated, info = (
                env.step(action)
            )

            total_reward += reward

        episode_rewards.append(total_reward)

        print(
            f"Episode {episode + 1}: "
            f"Reward = {total_reward:.2f}"
        )

    env.close()

    average_reward = sum(episode_rewards) / EPISODES
    successful_episodes = sum(
        reward >= 200 for reward in episode_rewards
    )

    print(f"\n{model_name} Results")
    print(f"Average Reward: {average_reward:.2f}")
    print(f"Successful Episodes: {successful_episodes}/{EPISODES}")

    return average_reward, successful_episodes


# Evaluate original PPO
original_results = evaluate_model(
    original_model,
    "Original PPO"
)

# Evaluate custom reward PPO
custom_results = evaluate_model(
    custom_model,
    "Custom Reward PPO"
)


# Compare results
print("\n========== FINAL COMPARISON ==========")

print(f"Original PPO Average: {original_results[0]:.2f}")
print(f"Custom PPO Average:   {custom_results[0]:.2f}")

print(f"Original PPO Successes: {original_results[1]}/{EPISODES}")
print(f"Custom PPO Successes:   {custom_results[1]}/{EPISODES}")

if custom_results[0] > original_results[0]:
    print("\nCustom Reward PPO performed better!")

elif original_results[0] > custom_results[0]:
    print("\nOriginal PPO performed better!")

else:
    print("\nBoth models achieved the same average reward.")
