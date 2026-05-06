from abc import ABC, abstractmethod

class BaseEnv(ABC):
    @abstractmethod
    def is_terminal(self, state) -> bool:
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, action):
        pass