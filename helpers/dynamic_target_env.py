from __future__ import annotations
from collections import defaultdict
from typing import Optional, Tuple
from helpers.base_env import BaseEnv, _perpendicular_actions, ACTIONS, ACTION_TO_DELTA

class DynamicTargetSlipperyGridWorld(BaseEnv):
    def __init__(
            self,
            rows: int = 4,
            cols: int = 4,
            start: Tuple[int, int] = (0, 0),
            target_start: Tuple[int, int] = (3, 3),
            target_move_prob: float = 0.8,
            slip_prob: float = 0.2,
            step_reward: float = -1.0,
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

        self.target_start = target_start
        self.target_move_prob = target_move_prob
        self.goal_reward = goal_reward
        self._target_row_column = target_start

        self.num_states = (rows * cols) ** 2
        self.grid_size = rows * cols

    def encode_state(self, ar: int, ac: int, tr: int, tc: int) -> int:
        agent_s = self.row_column_to_state(ar, ac)
        target_s = self.row_column_to_state(tr, tc)
        return agent_s * self.grid_size + target_s

    def decode_state(self, state: int) -> tuple[int, int, int, int]:
        agent_s, target_s = divmod(state, self.grid_size)
        ar, ac = divmod(agent_s, self.cols)
        tr, tc = divmod(target_s, self.cols)
        return ar, ac, tr, tc

    def reset(self, start: Optional[Tuple[int, int]] = None, target_start: Optional[Tuple[int, int]] = None) -> int:
        if start is not None:
            self.start_row_column = start
        if target_start is not None:
            self.target_start = target_start

        self._agent_row_column = self.start_row_column
        self._target_row_column = self.target_start
        self._steps = 0

        return self.encode_state(*self._agent_row_column, *self._target_row_column)

    def get_transition_distribution(self, state: int, action: int) -> list[tuple[float, int]]:
        assert action in ACTIONS
        ar, ac, tr, tc = self.decode_state(state)

        agent_probs = defaultdict(float)
        intended_ar, intended_ac = self._apply_action(ar, ac, action)
        agent_probs[(intended_ar, intended_ac)] += (1.0 - self.slip_prob)

        if self.slip_prob > 0.0:
            act_s_1, act_s_2 = _perpendicular_actions(action)
            r1, c1 = self._apply_action(ar, ac, act_s_1)
            r2, c2 = self._apply_action(ar, ac, act_s_2)
            agent_probs[(r1, c1)] += (self.slip_prob / 2.0)
            agent_probs[(r2, c2)] += (self.slip_prob / 2.0)

        target_probs = defaultdict(float)
        target_probs[(tr, tc)] += (1.0 - self.target_move_prob)

        if self.target_move_prob > 0.0:
            for a in ACTIONS:
                nr, nc = self._apply_action(tr, tc, a)
                target_probs[(nr, nc)] += (self.target_move_prob / 4.0)

        state_probs = defaultdict(float)
        for (nar, nac), p_a in agent_probs.items():
            for (ntr, ntc), p_t in target_probs.items():
                next_state = self.encode_state(nar, nac, ntr, ntc)
                state_probs[next_state] += p_a * p_t

        return [(p, s_next) for s_next, p in state_probs.items()]

    def is_terminal_state(self, state: int) -> bool:
        ar, ac, tr, tc = self.decode_state(state)
        return (ar, ac) == (tr, tc)

    def step(self, action: int) -> tuple[int, float, bool, dict]:
        assert action in ACTIONS
        self._steps += 1

        executed = self._sample_action_with_slip(action)
        ar, ac = self._agent_row_column
        nar, nac = self._apply_action(ar, ac, executed)

        tr, tc = self._target_row_column
        if self.rng.random() < self.target_move_prob:
            t_action = self.rng.choice(ACTIONS)
            ntr, ntc = self._apply_action(tr, tc, t_action)
        else:
            ntr, ntc = tr, tc

        self._agent_row_column = (nar, nac)
        self._target_row_column = (ntr, ntc)

        state = self.encode_state(ar, ac, tr, tc)
        next_state = self.encode_state(nar, nac, ntr, ntc)

        reward = self.reward(state, executed, next_state)
        done = self.is_terminal_state(next_state)

        if self.max_steps is not None and self._steps >= self.max_steps:
            done = True

        info = {"intended_action": action, "executed_action": executed, "steps": self._steps}
        return next_state, reward, done, info

    def reward(self, state: int, action: int, next_state: int) -> float:
        if self.is_terminal_state(state):
            return 0.0
        if self.is_terminal_state(next_state):
            return self.goal_reward
        return self.step_reward
