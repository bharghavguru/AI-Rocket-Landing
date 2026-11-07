
import gymnasium as gym


class CustomRewardWrapper(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.previous_potential = None
        self.gamma = 0.99

    def calculate_potential(self, observation):
        x = observation[0]
        vx = observation[2]
        vy = observation[3]
        angle = observation[4]
        angular_velocity = observation[5]

        potential = -(
            abs(x)
            + 0.5 * abs(vx)
            + 0.5 * abs(vy)
            + 0.3 * abs(angle)
            + 0.1 * abs(angular_velocity)
        )

        return potential

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)

        self.previous_potential = self.calculate_potential(
            observation
        )

        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = (
            self.env.step(action)
        )

        current_potential = self.calculate_potential(
            observation
        )

        if terminated or truncated:
            current_potential = 0

        shaping_reward = (
            self.gamma * current_potential
            - self.previous_potential
        )

        self.previous_potential = current_potential

        custom_reward = reward + shaping_reward

        return (
            observation,
            custom_reward,
            terminated,
            truncated,
            info
        )
