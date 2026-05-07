from collections import defaultdict
from typing import Optional, Tuple
from helpers.base_env import BaseEnv, _perpendicular_actions, ACTIONS, ACTION_TO_DELTA

class MultipleTargetSlipperyGridWorld(BaseEnv):

    def __init__(
            self,
            rows: int = 4,
            cols: int = 4,
            start: Tuple[int, int] = (0, 0),
            slip_prob: float = 0.2,
            step_reward: float = -1.0,
            max_steps: Optional[int] = None,
            seed: Optional[int] = None,
            goals: Optional[list[Tuple[int, int]]] = None,
            goal_rewards: Optional[list[float]] = None,
    ):
        super().__init__(
            rows=rows,
            cols=cols,
            start=start,
            slip_prob=slip_prob,
            step_reward=step_reward,
            max_steps=max_steps,
            seed=seed
        )

        self._goal_reward_map = None
        self._goals = goals if goals is not None else [(3, 3)]
        self._rewards = goal_rewards if goal_rewards is not None else [10.0]
        self.set_goals(self._goals, self._rewards)

    def reset(self, start: Optional[Tuple[int, int]] = None) -> int:
        if start is not None:
            self.start_row_column = start
        self._agent_row_column = self.start_row_column
        self._steps = 0
        return self.row_column_to_state(*self._agent_row_column)

    def get_transition_distribution(self, state: int, action: int) -> list[tuple[float, int]]:
        assert action in ACTIONS, f"Invalid action {action}. Use 0=U,1=R,2=D,3=L."
        probs = [0]*len(ACTION_TO_DELTA)
        probs[action] = 1 - self.slip_prob
        act_s_1, act_s_2 = _perpendicular_actions(action)
        probs[act_s_1] = self.slip_prob/2
        probs[act_s_2] = self.slip_prob/2
        state_probs = defaultdict(float)

        for a_real, prob in enumerate(probs):
            if prob == 0:
                continue

            next_state = self._apply_action_stateless(state, a_real)
            state_probs[next_state] += prob

        return [(p, s_next) for s_next, p in state_probs.items()]

    def is_terminal_state(self, state: int) -> bool:
        return state in self._goal_reward_map

    def step(self, action: int):
        assert action in ACTIONS
        self._steps += 1

        executed = self._sample_action_with_slip(action)

        r, c = self._agent_row_column
        self._agent_row_column = self._apply_action(r, c, executed)

        next_state = self.row_column_to_state(*self._agent_row_column)

        done = next_state in self._goal_reward_map
        if self.max_steps is not None and self._steps >= self.max_steps:
            done = True

        reward = self._goal_reward_map.get(next_state, self.step_reward)

        info = {"intended_action": action, "executed_action": executed, "steps": self._steps}
        return next_state, reward, done, info

    def set_goals(self, goals: list[Tuple[int, int]], goal_rewards: list[float]) -> None:
        assert len(goals) == len(goal_rewards)
        self._goal_reward_map = {
            self.row_column_to_state(r, c): reward
            for (r, c), reward in zip(goals, goal_rewards)
        }

    def reward(self, state: int, action: int, next_state: int) -> float:
        if state in self._goal_reward_map:
            return 0.0
        return self._goal_reward_map.get(next_state, self.step_reward)