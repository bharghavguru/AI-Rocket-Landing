
# 🚀 AI Rocket Landing Simulator

An AI-powered rocket landing simulation using
Reinforcement Learning and Proximal Policy Optimization (PPO).

## About the Project

This project trains an AI agent to land a rocket safely
on a landing platform using the Lunar Lander environment.

The agent learns through trial and error by interacting
with the environment and receiving rewards or penalties.

## Technologies Used

- Python
- Gymnasium
- Stable-Baselines3
- PPO (Proximal Policy Optimization)
- PyTorch
- Box2D

## Project Features

- Lunar Lander physics simulation
- Random agent baseline
- PPO-based reinforcement learning
- Model training and saving
- Model evaluation over multiple episodes
- Custom reward shaping experiments
- Reward hacking analysis

## Project Structure

```text
AI-Rocket-Landing/
├── main.py
├── train.py
├── evaluate.py
├── record_video.py
├── requirements.txt
├── models/
└── videos/
```

## Installation

Clone the repository:

```bash
git clone YOUR_REPOSITORY_URL
cd AI-Rocket-Landing
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
.\venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Simulation

```bash
python main.py
```

## Train the AI

```bash
python train.py
```

## Evaluate the Model

```bash
python evaluate.py
```

## Future Improvements

- Experiment with custom reward functions
- Study reward hacking
- Improve landing success rate
- Compare different training configurations

## Project Status

Work in Progress 🚧

## Inspiration

Inspired by the Lunar Lander reinforcement learning
environment and reward shaping experiments.

## Author

P.G.Gurubharghav

GitHub: https://github.com/bharghavguru
