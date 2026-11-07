
import matplotlib.pyplot as plt
import os

# Actual results from your experiment
original_rewards = [
    275.93, 191.35, 230.91, 231.75, 242.53,
    258.34, 242.88, 242.59, 251.98, 279.15,
    246.08, 252.90, 217.20, 233.24, 248.86,
    262.14, 188.14, 50.71, 288.74, 234.12
]

custom_rewards = [
    217.66, 147.97, 150.76, 148.04, 179.24,
    199.75, 212.30, 157.09, -43.29, -64.13,
    198.53, 207.12, 187.56, -60.70, 209.01,
    197.49, 172.57, 200.13, 245.52, 153.27
]

original_average = sum(original_rewards) / 20
custom_average = sum(custom_rewards) / 20

original_success = sum(r >= 200 for r in original_rewards)
custom_success = sum(r >= 200 for r in custom_rewards)

os.makedirs("results", exist_ok=True)

# Graph 1: Average reward comparison
plt.figure(figsize=(8, 5))

plt.bar(
    ["Original PPO", "Custom Reward PPO"],
    [original_average, custom_average]
)

plt.ylabel("Average Episode Reward")
plt.title("Original PPO vs Custom Reward PPO")
plt.savefig("results/average_reward.png", dpi=300)
plt.show()

# Graph 2: Episode-by-episode comparison
plt.figure(figsize=(10, 5))

plt.plot(
    range(1, 21),
    original_rewards,
    marker="o",
    label="Original PPO"
)

plt.plot(
    range(1, 21),
    custom_rewards,
    marker="s",
    label="Custom Reward PPO"
)

plt.axhline(200, linestyle="--", label="200 Reward Benchmark")

plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Episode Reward Comparison")
plt.legend()
plt.grid(True)

plt.savefig("results/episode_comparison.png", dpi=300)
plt.show()

print("Graphs saved inside results folder!")
