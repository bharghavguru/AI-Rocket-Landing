
import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np

EPISODES = 20

# Load all three models
models = {
    "Original PPO": PPO.load("models/rocket_ppo"),
    "Custom Reward V1": PPO.load("models/rocket_ppo_custom"),
    "Custom Reward V2": PPO.load("models/rocket_ppo_custom_v2")
}


def evaluate_model(model, model_name):

    env = gym.make("LunarLander-v3")
    rewards = []

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

        rewards.append(total_reward)

        print(
            f"Episode {episode + 1}: "
            f"Reward = {total_reward:.2f}"
        )

    env.close()

    average = np.mean(rewards)
    std_dev = np.std(rewards)
    benchmark_count = sum(r >= 200 for r in rewards)

    print(f"\n{model_name} Results")
    print(f"Average Reward: {average:.2f}")
    print(f"Standard Deviation: {std_dev:.2f}")
    print(f"Episodes >= 200: {benchmark_count}/{EPISODES}")

    return {
        "average": average,
        "std": std_dev,
        "benchmark": benchmark_count,
        "rewards": rewards
    }


results = {}

for name, model in models.items():
    results[name] = evaluate_model(model, name)


print("\n========== FINAL MODEL COMPARISON ==========")

for name, result in results.items():

    print(f"\n{name}")
    print(f"Average Reward: {result['average']:.2f}")
    print(f"Standard Deviation: {result['std']:.2f}")
    print(f"Episodes >= 200: {result['benchmark']}/{EPISODES}")

best_model = max(
    results,
    key=lambda name: results[name]["average"]
)

print(f"\nBest average reward: {best_model}")

env.close() if False else None
