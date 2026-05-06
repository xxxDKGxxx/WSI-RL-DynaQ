import random
import numpy as np
from helpers.env import SlipperyGridWorld


class DynaQAgent:
    def __init__(
        self,
        num_episodes: int,
        num_sim_iter: int,
        eps: float,
        eps_min: float,
        eps_decay: float,
        alpha: float,
        gamma: float,
        env: SlipperyGridWorld,
        num_states: int,
        actions: list[int],
    ) -> None:
        self.num_episodes = num_episodes
        self.num_sim_iter = num_sim_iter
        self.eps = eps
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.alpha = alpha
        self.gamma = gamma
        self.env = env
        self.num_states = num_states
        self.actions = actions
        self.num_actions = len(actions)
        self.reset()

    def reset(self) -> None:
        self.Q = np.zeros((self.num_states, self.num_actions))
        self.model: dict[tuple[int, int], tuple[int, int]] = {}

    def run(self) -> None:
        epsilon = self.eps

        for episode_num in range(self.num_episodes):
            print(
                f"\rProgress: {(episode_num / self.num_episodes * 100):6.2f}%", end=""
            )

            state = self.env.reset()
            done = False

            while not done:
                action = self._epsilon_greedy_policy(self.Q, state, epsilon)
                next_state, reward, done, _ = self.env.step(action)
                self.Q[state, action] = self.Q[state, action] + self.alpha * (
                    reward
                    + (1 - done) * self.gamma * np.max(self.Q[next_state])
                    - self.Q[state, action]
                )
                state = next_state

                self.run_simulation()

            epsilon = max(self.eps_min, epsilon * self.eps_decay)

    def run_simulation(self) -> None:
        # TODO implement model gathering and dynaq simulation
        pass

    def _epsilon_greedy_policy(self, Q, state, epsilon) -> int:
        if random.random() < epsilon:
            return random.choice(self.actions)

        return int(np.argmax(Q[state]))
