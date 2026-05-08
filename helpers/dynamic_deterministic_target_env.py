from __future__ import annotations
from collections import defaultdict
from typing import Optional, Tuple
from helpers.base_env import BaseEnv, _perpendicular_actions, ACTIONS, ACTION_TO_DELTA


class DynamicDeterministicTargetSlipperyGridWorld(BaseEnv):
    def __init__(
        self,
        rows: int = 4,
        cols: int = 4,
        start: Tuple[int, int] = (0, 0),
        target_start: Tuple[int, int] = (3, 3),
        target_start_dir: int = 1,  # 1 dla ruchu w prawo, -1 dla ruchu w lewo
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
            seed=seed,
        )

        self.target_start = target_start
        self.target_start_dir = target_start_dir
        self.goal_reward = goal_reward
        self._target_row_column = target_start
        self._target_dir = target_start_dir

        self.grid_size = rows * cols
        # Przestrzeń stanów powiększona o 2 (kierunek celu: w prawo lub w lewo)
        self.num_states = ((rows * cols) ** 2) * 2

    def encode_state(self, ar: int, ac: int, tr: int, tc: int, t_dir: int) -> int:
        agent_s = self.row_column_to_state(ar, ac)
        target_s = self.row_column_to_state(tr, tc)
        # Kodujemy kierunek: 0 dla w prawo (1), 1 dla w lewo (-1)
        t_dir_idx = 0 if t_dir == 1 else 1
        return (agent_s * self.grid_size + target_s) * 2 + t_dir_idx

    def decode_state(self, state: int) -> tuple[int, int, int, int, int]:
        state, t_dir_idx = divmod(state, 2)
        t_dir = 1 if t_dir_idx == 0 else -1
        agent_s, target_s = divmod(state, self.grid_size)
        ar, ac = divmod(agent_s, self.cols)
        tr, tc = divmod(target_s, self.cols)
        return ar, ac, tr, tc, t_dir

    def reset(
        self,
        start: Optional[Tuple[int, int]] = None,
        target_start: Optional[Tuple[int, int]] = None,
    ) -> int:
        if start is not None:
            self.start_row_column = start
        if target_start is not None:
            self.target_start = target_start

        self._agent_row_column = self.start_row_column
        self._target_row_column = self.target_start
        self._target_dir = self.target_start_dir
        self._steps = 0

        return self.encode_state(
            *self._agent_row_column, *self._target_row_column, self._target_dir
        )

    def _get_next_target_pos(
        self, tr: int, tc: int, t_dir: int
    ) -> tuple[int, int, int]:
        """Oblicza następną pozycję celu deterministycznie odbijając się od ścian."""
        ntc = tc + t_dir
        nt_dir = t_dir

        if ntc >= self.cols:
            ntc = max(0, self.cols - 2)
            nt_dir = -1
        elif ntc < 0:
            ntc = min(self.cols - 1, 1)
            nt_dir = 1

        return tr, ntc, nt_dir

    def get_transition_distribution(
        self, state: int, action: int
    ) -> list[tuple[float, int]]:
        assert action in ACTIONS
        ar, ac, tr, tc, t_dir = self.decode_state(state)

        agent_probs = defaultdict(float)
        intended_ar, intended_ac = self._apply_action(ar, ac, action)
        agent_probs[(intended_ar, intended_ac)] += 1.0 - self.slip_prob

        if self.slip_prob > 0.0:
            act_s_1, act_s_2 = _perpendicular_actions(action)
            r1, c1 = self._apply_action(ar, ac, act_s_1)
            r2, c2 = self._apply_action(ar, ac, act_s_2)
            agent_probs[(r1, c1)] += self.slip_prob / 2.0
            agent_probs[(r2, c2)] += self.slip_prob / 2.0

        # Deterministyczny ruch celu
        ntr, ntc, nt_dir = self._get_next_target_pos(tr, tc, t_dir)

        state_probs = defaultdict(float)
        for (nar, nac), p_a in agent_probs.items():
            next_state = self.encode_state(nar, nac, ntr, ntc, nt_dir)
            state_probs[next_state] += p_a

        return [(p, s_next) for s_next, p in state_probs.items()]

    def is_terminal_state(self, state: int) -> bool:
        ar, ac, tr, tc, _ = self.decode_state(state)
        return (ar, ac) == (tr, tc)

    def step(self, action: int) -> tuple[int, float, bool, dict]:
        assert action in ACTIONS
        self._steps += 1

        executed = self._sample_action_with_slip(action)
        ar, ac = self._agent_row_column
        nar, nac = self._apply_action(ar, ac, executed)

        tr, tc = self._target_row_column
        t_dir = self._target_dir

        # Obliczenie nowego położenia dla celu
        ntr, ntc, nt_dir = self._get_next_target_pos(tr, tc, t_dir)

        self._agent_row_column = (nar, nac)
        self._target_row_column = (ntr, ntc)
        self._target_dir = nt_dir

        state = self.encode_state(ar, ac, tr, tc, t_dir)
        next_state = self.encode_state(nar, nac, ntr, ntc, nt_dir)

        reward = self.reward(state, executed, next_state)
        done = self.is_terminal_state(next_state)

        if self.max_steps is not None and self._steps >= self.max_steps:
            done = True

        info = {
            "intended_action": action,
            "executed_action": executed,
            "steps": self._steps,
        }
        return next_state, reward, done, info

    def reward(self, state: int, action: int, next_state: int) -> float:
        if self.is_terminal_state(state):
            return 0.0
        if self.is_terminal_state(next_state):
            return self.goal_reward
        return self.step_reward
