import random
import numpy as np
from helpers.base_env import BaseEnv


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
        env: BaseEnv,
        num_states: int,
        actions: list[int],
        planning_steps: int,
        kappa: float,
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
        self.planning_steps = planning_steps
        self.kappa = kappa
        self.reset()

    def reset(self) -> None:
        self.Q = np.zeros((self.num_states, self.num_actions))
        self.model: dict[tuple[int, int], tuple[int, int]] = {}
        self.last_visited = np.zeros((self.num_states, self.num_actions))
        self.current_step = 0

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
                self.last_visited[state, action] = self.current_step
                next_state, reward, done, _ = self.env.step(action)
                self.Q[state, action] = self.Q[state, action] + self.alpha * (
                    reward
                    + (1 - done) * self.gamma * np.max(self.Q[next_state])
                    - self.Q[state, action]
                )
                state = next_state

                self.run_simulation(state, action, reward, next_state)

            epsilon = max(self.eps_min, epsilon * self.eps_decay)

    def run_simulation(self, state, action, reward, next_state) -> None:
        # TODO implement model gathering and dynaq simulation
        self.model[state, action] = reward, next_state
        self.planning_plus()
        self.current_step += 1

    def planning(self):
        visited_state_actions = list(self.model.keys())

        for i in range(self.planning_steps):
            done = False
            state, action = random.choice(visited_state_actions)
            reward, next_state = self.model[state, action]

            if self.env.is_terminal_state(next_state):
                done = True

            self.Q[state, action] = self.Q[state, action] + self.alpha * (
                    reward
                    + (1 - done) * self.gamma * np.max(self.Q[next_state])
                    - self.Q[state, action]
            )

    def planning_plus(self):
        visited_state_actions = list(self.model.keys())

        for i in range(self.planning_steps):
            done = False
            state, action = random.choice(visited_state_actions)
            action = random.choice(self.actions)

            if (state, action) in self.model:
                reward, next_state = self.model[state, action]
            else:
                reward = 0.0
                next_state = state

            reward += self.kappa * np.sqrt(self.current_step - self.last_visited[state, action])

            if self.env.is_terminal_state(next_state):
                done = True

            self.Q[state, action] = self.Q[state, action] + self.alpha * (
                    reward
                    + (1 - done) * self.gamma * np.max(self.Q[next_state])
                    - self.Q[state, action]
            )

    def _epsilon_greedy_policy(self, Q, state, epsilon) -> int:
        if random.random() < epsilon:
            return random.choice(self.actions)

        return int(np.argmax(Q[state]))
