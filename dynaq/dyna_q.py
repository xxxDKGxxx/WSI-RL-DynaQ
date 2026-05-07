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
        self.model = StochasticDynaQModel()

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

                self.model.update(state, action, reward, next_state, done)

                state = next_state

                self.run_simulation()

            epsilon = max(self.eps_min, epsilon * self.eps_decay)

    def run_simulation(self) -> None:
        for _ in range(self.num_sim_iter):
            state, action, reward, next_state, done = self.model.sample()
            self.Q[state, action] = self.Q[state, action] + self.alpha * (
                reward
                + (1 - done) * self.gamma * np.max(self.Q[next_state])
                - self.Q[state, action]
            )

    def _epsilon_greedy_policy(self, Q, state, epsilon) -> int:
        if random.random() < epsilon:
            return random.choice(self.actions)

        return int(np.argmax(Q[state]))


class StochasticDynaQModel:
    def __init__(self) -> None:
        self.model = {}

    def update(self, state, action, reward, next_state, done) -> None:
        if state not in self.model:
            self.model[state] = {}
        if action not in self.model[state]:
            self.model[state][action] = {}
        if next_state not in self.model[state][action]:
            self.model[state][action][next_state] = {
                "count": 0,
                "reward_sum": 0.0,
                "done": done,
            }

        self.model[state][action][next_state]["count"] += 1
        self.model[state][action][next_state]["reward_sum"] += reward

    def sample(self) -> tuple[int, int, float, int, bool]:
        state = random.choice(list(self.model.keys()))
        action = random.choice(list(self.model[state].keys()))

        transitions = self.model[state][action]
        next_states = list(transitions.keys())

        counts = [transitions[ns]["count"] for ns in next_states]

        sampled_next_state = random.choices(next_states, weights=counts, k=1)[0]

        stats = transitions[sampled_next_state]
        expected_reward = stats["reward_sum"] / stats["count"]
        done = stats["done"]

        return state, action, expected_reward, sampled_next_state, done
