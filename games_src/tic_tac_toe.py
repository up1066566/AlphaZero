from games_src.game import BoardGame as BG
import numpy as np
import torch

class TicTacToe(BG):
    def __init__(self):
        super().__init__(rows = 3, cols = 3, action_channels = 1)
        
    def get_initial_state(self) -> np.ndarray:
        return np.zeros((self.rows, self.cols), dtype = np.int8)
    
    def get_next_state(self, state: np.ndarray, action: int, player: int) -> np.ndarray:
        next_state = state.copy()
        row = action//3
        col = action%3
        next_state[row][col] = player
        return next_state
        
    def get_valid_actions(self, state: np.ndarray) -> np.ndarray:
        return (state.reshape(-1)==0).astype(np.uint8)
    
    def get_result_and_terminated(self, state: np.ndarray, action: int) -> tuple[float, int]:
        row = action // 3
        col = action % 3
        player = state[row, col]
        
        #check for row or column:
        if np.all(state[row, :] == player) or  np.all(state[:, col]==player):
            return player, 1
        
        #check for diagonals:
        if row == col and np.all(np.diag(state) == player):
            return player, 1
        if (row+col == 2) and np.all(np.diag(np.fliplr(state)) == player):
            return player, 1
        # terminated with draw
        if not self.get_valid_actions(state):
            return 0, 1 
        
        return 0,0
    
    def get_encoded_state(self, state: np.ndarray) -> torch.Tensor:
        
        # returns a feature map with spatial dimension 3x3 and 2 channels 
        # 1 channel for player 1 the player who plays now, 1 channel for the opponent
        
        current_player = 1 if np.sum(state) == 0 else -1
        
        encoded = np.stack([
            state == current_player,
            state == -current_player,
        ]).astype(np.float32)
        
        return torch.tensor(encoded)  # Shape: (2, 3, 3) 