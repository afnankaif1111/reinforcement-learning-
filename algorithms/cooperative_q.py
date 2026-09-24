"""
Cooperative Shared-Map Q-Learning for Swarm Robots.
Coordinates the swarm through:
1. Shared Fleet Value Function (Federated / Centralized Q-table):
   Agents pool experiences into a shared policy, accelerating convergence.
2. Team Reward Sharing (Credit Assignment):
   Blends individual reward with the swarm collective discovery reward.
"""

import numpy as np
from typing import List
from config import SwarmConfig, NUM_ACTIONS

class CooperativeQLearning:
    """
    Cooperative Multi-Agent Q-Learning.
    Uses shared policy representation and collective team reward shaping.
    """
    def __init__(self, num_robots: int, num_states: int = 320, num_actions: int = NUM_ACTIONS, config: SwarmConfig = None, team_coop_weight: float = 0.3):
        self.num_robots = num_robots
        self.num_states = num_states
        self.num_actions = num_actions
        self.cfg = config or SwarmConfig()
        self.team_weight = team_coop_weight  # Weight for collective team reward blending
        
        self.lr = self.cfg.learning_rate
        self.gamma = self.cfg.discount_factor
        self.epsilon = self.cfg.epsilon_start
        self.epsilon_min = self.cfg.epsilon_min
        self.epsilon_decay = self.cfg.epsilon_decay
        
        # Shared fleet Q-table: shape (num_states, num_actions)
        # All robots share and refine this collective brain
        self.shared_q_table = np.zeros((self.num_states, self.num_actions), dtype=np.float32)

    def select_action(self, agent_id: int, state: int, training: bool = True) -> int:
        """Epsilon-greedy action selection using the shared swarm policy."""
        if training and np.random.rand() < self.epsilon:
            return int(np.random.randint(0, self.num_actions))
        
        q_vals = self.shared_q_table[state]
        max_val = np.max(q_vals)
        best_actions = np.where(q_vals == max_val)[0]
        return int(np.random.choice(best_actions))

    def select_actions(self, states: List[int], training: bool = True) -> List[int]:
        """Selects actions for all swarm agents."""
        return [self.select_action(i, states[i], training=training) for i in range(self.num_robots)]

    def update_team(self, states: List[int], actions: List[int], raw_rewards: List[float], next_states: List[int], done: bool):
        """
        Applies Team Reward Sharing and updates the shared Q-table with the swarm's transitions.
        
        Reward blending:
            r_coop_i = (1 - team_weight) * r_local_i + team_weight * mean(rewards)
        """
        mean_team_reward = float(np.mean(raw_rewards))
        
        for i in range(self.num_robots):
            s = states[i]
            a = actions[i]
            # Blend individual reward with team progress
            blended_reward = (1.0 - self.team_weight) * raw_rewards[i] + self.team_weight * mean_team_reward
            next_s = next_states[i]
            
            current_q = self.shared_q_table[s, a]
            if done:
                target = blended_reward
            else:
                best_next_q = np.max(self.shared_q_table[next_s])
                target = blended_reward + self.gamma * best_next_q
                
            td_error = target - current_q
            self.shared_q_table[s, a] += self.lr * td_error

    def step_decay(self):
        """Decays epsilon after each episode."""
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
    def get_stats(self) -> dict:
        """Returns summary statistics for logging."""
        return {
            "epsilon": float(self.epsilon),
            "mean_q_val": float(np.mean(self.shared_q_table)),
            "max_q_val": float(np.max(self.shared_q_table)),
            "non_zero_q_entries": int(np.count_nonzero(self.shared_q_table))
        }
