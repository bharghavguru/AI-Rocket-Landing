
import matplotlib.pyplot as plt
import os

# Evaluation results from all three models
original = [
    275.93, 191.35, 230.91, 231.75, 242.53,
    258.34, 242.88, 242.59, 251.98, 279.15,
    246.08, 252.90, 217.20, 233.24, 248.86,
    262.14, 188.14, 50.71, 288.74, 234.12
]

v1 = [
    217.66, 147.97, 150.76, 148.04, 179.24,
    199.75, 212.30, 157.09, -43.29, -64.13,
    198.53, 207.12, 187.56, -60.70, 209.01,
    197.49, 172.57, 200.13, 245.52, 153.27
]

v2 = [
    241.45, -14.91, 212.65, 211.03, 229.99,
    239.74, 252.66, 231.17, 232.49, 208.80,
    216.96, 226.04, 237.14, -3.49, 219.69,
    248.97, 205.02, -5.85, 238.77, 167.29
]

os.makedirs("results", exist_ok=True)

models = ["Original PPO", "Custom Reward V1", "Custom Reward V2"]

averages = [
    sum(original) / 20,
    sum(v1) / 20,
    sum(v2) / 20
]

successes = [
    sum(r >= 200 for r in original),
    sum(r >= 200 for r in v1),
    sum(r >= 200 for r in v2)
]

# Graph 1: Average reward
plt.figure(figsize=(9, 5))
plt.bar(models, averages)

plt.ylabel("Average Episode Reward")
plt.title("Three-Model Average Reward Comparison")
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("results/three_model_average_reward.png", dpi=300)
plt.show()

# Graph 2: Episode-wise comparison
plt.figure(figsize=(11, 6))

episodes = range(1, 21)

plt.plot(episodes, original, marker="o", label="Original PPO")
plt.plot(episodes, v1, marker="s", label="Custom Reward V1")
plt.plot(episodes, v2, marker="^", label="Custom Reward V2")

plt.axhline(200, linestyle="--", label="200 Reward Benchmark")

plt.xlabel("Episode")
plt.ylabel("Episode Reward")
plt.title("Episode-wise Performance Comparison")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("results/three_model_episode_comparison.png", dpi=300)
plt.show()

# Graph 3: Benchmark episodes
plt.figure(figsize=(9, 5))
plt.bar(models, successes)

plt.ylabel("Episodes with Reward >= 200")
plt.title("Benchmark Episode Comparison")
plt.ylim(0, 20)
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("results/benchmark_comparison.png", dpi=300)
plt.show()

print("All three comparison graphs saved in results/")
