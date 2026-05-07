from abc import ABC, abstractmethod
from collections import defaultdict
import random
from typing import Optional, Tuple


# Action mapping: 0=Up, 1=Right, 2=Down, 3=Left
ACTIONS = (0, 1, 2, 3)
ACTION_TO_DELTA = {
    0: (-1, 0),
    1: (0, +1),
    2: (+1, 0),
    3: (0, -1),
}


def _perpendicular_actions(a: int) -> Tuple[int, int]:
    if a in (0, 2):   # Up or Down
        return (3, 1) # Left, Right
    else:             # Left or Right
        return (0, 2) # Up, Down


class BaseEnv:

    def __init__(
            self,
            rows: int = 4,
            cols: int = 4,
            start: Tuple[int, int] = (0, 0),
            slip_prob: float = 0.2,
            step_reward: float = -1.0,
            max_steps: Optional[int] = None,
            seed: Optional[int] = None,
    ):
        assert rows > 0 and cols > 0
        assert 0.0 <= slip_prob <= 1.0

        self.rows = rows
        self.cols = cols
        self.start_row_column = start
        self.slip_prob = slip_prob
        self.step_reward = step_reward
        self.max_steps = max_steps

        self.rng = random.Random(seed)
        self._steps = 0
        self._agent_row_column = start

        self.num_states = rows * cols
        self.nA = len(ACTIONS)

    @abstractmethod
    def is_terminal_state(self, state) -> bool:
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, action):
        pass

    # --- helpers ---
    def row_column_to_state(self, r: int, c: int) -> int:
        return r * self.cols + c

    def state_to_row_column(self, s: int) -> Tuple[int, int]:
        return divmod(s, self.cols)

    def _in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.rows and 0 <= c < self.cols

    def _apply_action(self, r: int, c: int, a: int) -> Tuple[int, int]:
        dr, dc = ACTION_TO_DELTA[a]
        nr, nc = r + dr, c + dc
        if self._in_bounds(nr, nc):
            return nr, nc
        return r, c

    def _sample_action_with_slip(self, intended: int) -> int:
        if self.rng.random() >= self.slip_prob:
            return intended
        left, right = _perpendicular_actions(intended)
        return left if self.rng.random() < 0.5 else right

    def _apply_action_stateless(self, state: int, action: int):
        r, c = self.state_to_row_column(state)
        dr, dc = ACTION_TO_DELTA[action]
        nr, nc = r + dr, c + dc
        if self._in_bounds(nr, nc):
            return self.row_column_to_state(nr, nc)
        return self.row_column_to_state(r, c)
