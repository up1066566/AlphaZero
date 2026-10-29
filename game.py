from abc import ABC, abstractmethod
import numpy as np
import torch

class Boardgame:
    def __init__(self, rows: int, cols: int, state_channels: int =1, action_channels: int=1):
        self.rows = rows
        self.cols = cols
        self.state_channels = action_channels
        self.action_channels = action_channels
        self.player = 1
        self.state = None
    
    @abstractmethod
    def set_initial_state(self) -> None:
        pass
    
    def get_state(self) -> np.ndarray:
        return self.state
    
    @abstractmethod
    def get_actions(self) -> np.ndarray:
        "It must return the action space rows*cols*action_channels"
        pass
    
    @abstractmethod
    def get_policy(self) -> np.ndarray:
        "It will be a vector, with all valid actions and their probabilities"
        pass
    
    @abstractmethod
    def get_next_state(self, action) -> np.ndarray:
        pass
    
    def switch_player(self) -> None:
        self.player = -self.player
    
    def get_opp_state(self) -> np.ndarray:
        return -self.state