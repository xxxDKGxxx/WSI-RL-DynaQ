from __future__ import annotations
from collections import defaultdict
from typing import Optional, Tuple
from helpers.base_env import BaseEnv, _perpendicular_actions, ACTIONS, ACTION_TO_DELTA


class CliffSlipperyGridWorld(BaseEnv):
    def __init__(
            self,
            rows: int = 4,
            cols: int = 12,
            start: Tuple[int, int] = (3, 0),
            goal: Tuple[int, int] = (3, 11),
            cliff: Optional[list[Tuple[int, int]]] = None,
            slip_prob: float = 0.0,
            step_reward: float = -1.0,
            cliff_reward: float = -100.0,
            goal_reward: float = 10.0,
            max_steps: Optional[int] = None,
            seed: Optional[int] = None,
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

        self.goal_state = self.row_column_to_state(*goal)
        self.cliff_reward = cliff_reward
        self.goal_reward = goal_reward

        if cliff is None:
            self.cliff_states = {self.row_column_to_state(3, i) for i in range(1, 11)}
        else:
            self.cliff_states = {self.row_column_to_state(r, c) for r, c in cliff}

    def reset(self, start: Optional[Tuple[int, int]] = None) -> int:
        if start is not None:
            self.start_row_column = start
        self._agent_row_column = self.start_row_column
        self._steps = 0
        return self.row_column_to_state(*self._agent_row_column)

    def get_transition_distribution(self, state: int, action: int) -> list[tuple[float, int]]:
        assert action in ACTIONS

        state_probs = defaultdict(float)

        next_state_intended = self._apply_action_stateless(state, action)
        if next_state_intended in self.cliff_states:
            next_state_intended = self.row_column_to_state(*self.start_row_column)
        state_probs[next_state_intended] += (1.0 - self.slip_prob)

        if self.slip_prob > 0.0:
            act_s_1, act_s_2 = _perpendicular_actions(action)

            next_state_1 = self._apply_action_stateless(state, act_s_1)
            if next_state_1 in self.cliff_states:
                next_state_1 = self.row_column_to_state(*self.start_row_column)

            next_state_2 = self._apply_action_stateless(state, act_s_2)
            if next_state_2 in self.cliff_states:
                next_state_2 = self.row_column_to_state(*self.start_row_column)

            state_probs[next_state_1] += (self.slip_prob / 2.0)
            state_probs[next_state_2] += (self.slip_prob / 2.0)

        return list(state_probs.items())

    def is_terminal_state(self, state: int) -> bool:
        return state == self.goal_state

    def step(self, action: int):
        assert action in ACTIONS, f"Invalid action {action}. Use 0=U,1=R,2=D,3=L."
        self._steps += 1

        intended = action
        executed = self._sample_action_with_slip(intended)

        r, c = self._agent_row_column
        nr, nc = self._apply_action(r, c, executed)
        state = self.row_column_to_state(r, c)
        next_state = self.row_column_to_state(nr, nc)

        reward = self.reward(state, next_state)

        if next_state in self.cliff_states:
            self._agent_row_column = self.start_row_column
            next_state = self.row_column_to_state(*self.start_row_column)
            done = False
        else:
            self._agent_row_column = (nr, nc)
            done = (next_state == self.goal_state)

        if self.max_steps is not None and self._steps >= self.max_steps:
            done = True


        info = {"intended_action": intended, "executed_action": executed, "steps": self._steps}
        return next_state, reward, done, info

    def reward(self, state: int, next_state: int) -> float:
        if state == self.goal_state:
            return 0.0

        if next_state in self.cliff_states:
            return self.cliff_reward

        if next_state == self.goal_state:
            return self.goal_reward

        return self.step_reward