from abc import ABC, abstractmethod
import numpy as np
import torch

class Boardgame(ABC):
    def __init__(self, rows: int, cols: int, action_channels: int = 1):
        self.rows = rows
        self.cols = cols
        self.action_channels = action_channels
        self.action_size = self.rows * self.cols * self.action_channels

    @abstractmethod
    def get_initial_state(self) -> np.ndarray:
        """Return initial state of size rows*cols*channels"""
        pass

    @abstractmethod
    def get_next_state(self, state: np.ndarray, action: int, player: int) -> np.ndarray:
        """Returns next state, after the action of player 1 or -1"""
        pass

    @abstractmethod
    def get_valid_actions(self, state: np.ndarray) -> np.ndarray:
        """
        Returns 1D binary array with length action_size
        valid actions are True, invalid values are False
        """
        pass

    @abstractmethod
    def get_result_and_terminated(self, state: np.ndarray, action: int) -> tuple[float, bool]:
        """
        Checks if the game is terminated and gets the result
        - result: 1.0 (win), 0.0 (draw / not terminated), -1.0 (lose)
        """
        pass

    @abstractmethod
    def get_encoded_state(self, state: np.ndarray) -> torch.Tensor:
        """Converts state to PyTorch Tensor with shape (state_channels, rows, cols)."""
        pass

    def switch_player(self, player: int) -> int:
        """Return opponent player (1 -> -1, -1 -> 1)."""
        return -player