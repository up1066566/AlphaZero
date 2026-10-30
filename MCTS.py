import numpy as np
import torch
import random
from game import BoardGame as BG

# PUCT
class MCTS:
    def __init__(self):
        self.c_base = 19562
        self.c_init = 2.5
        self.tree = {}          # state: [actions]
        self.states_count = {}  # state: N(s)
        self.states_actions_count = {}  # (state,action): N(s,a)
        self.Q_values = {}            # (state, action): Q(s,a) 
    
    def get_N(self, state: torch.Tensor) -> int:
        if state in self.states_count:
            return self.states_count[state]
        return 0
    
    def get_N(self, state: torch.Tensor, action: torch.Tensor) -> int:
        if (state, action) in self.states_actions_count:
            return self.states_actions_count[(state,action)]
        return 0
    
    def get_Q(self, state, action: torch.Tensor) -> float:
        if (state, action) in self.Q_values:
            return self.Q_values[(state, action)]
        return -1
    
    def update_N(self, state) -> None:
        self.states_count[state] = self.get_N(state)+1
    
    def update_N(self, state: torch.Tensor, action: torch.Tensor) -> None:
        self.states_count[(state, action)] = self.get_N(state, action) + 1
    
    # it must be applied after update_N(state, action), to avoid division by 0
    def update_Q(self, state: torch.Tensor, action: torch.Tensor, v_estimation: float) -> None:
        q = self.get_Q(state,action) 
        self.Q_values[(state, action)] = q + (v_estimation - q)/self.get_N(state, action)
    
    # We expand the tree only if we go to a state that is not in tree
    def expand_tree(self, state: torch.Tensor, valid_actions: np.ndarray, game: BG)->None:
        self.tree[state] = []
        self.states_count[state] = 1
        for action in valid_actions:
            self.tree[state].append(action)
            self.states_actions_count[(state, action)] = 0
            self.Q_values[(state,action)] = -1
    
    def get_c(self, state: torch.Tensor) -> float:
        return np.log((self.get_N(state)+self.c_base+1)/self.c_base)+self.c_init
    
    def get_U(self, state: torch.Tensor, action: torch.Tensor, prior_prob) -> float:
        return self.get_c*prior_prob*(np.sqrt(self.get_N(state))/(1+self.get_N(state, action)))
    
    def objective_function(self, state: torch.Tensor, action: torch.Tensor, prior_prob) -> float:
        return self.get_Q(state, action) + self.get_U(state, action, prior_prob)
    
    def select_action(self, state: torch.Tensor) -> int:
        actions = self.tree[state]
        best_action = actions[0]
        best_actions = []
        best_objective = self.objective_function(state, best_action)
        for action in actions:
            objective = self.objective_function(state, action)
            if objective > best_objective:
                best_objective = objective
                best_actions = [action]
            elif objective == best_objective:
                best_actions.append[action]
        return random.choice(best_actions)