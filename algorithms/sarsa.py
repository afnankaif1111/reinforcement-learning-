"""
SARSA (State-Action-Reward-State-Action) for Swarm Robots.
An on-policy Temporal Difference (TD) control algorithm.

Update rule:
    Q_i(s, a) <- Q_i(s, a) + alpha * [r + gamma * Q_i(s', a') - Q_i(s, a)]
    where a' is the action actually chosen by the epsilon-greedy policy at state s'.

Key Pedagogical Property:
SARSA is safer than Q-learning around hazards because it accounts for its own
probabilistic exploratory errors (epsilon-greedy slips into hazards).
"""

import numpy as np
from typing import List
from config import SwarmConfig, NUM_ACTIONS

class SARSALearning:
    """
    On-policy SARSA agent fleet.
    Learns conservative, hazard-sensitive navigation policies.
    """
    def __init__(self, num_robots: int, num_states: int = 320, num_actions: int = NUM_ACTIONS, config: SwarmConfig = None):
        self.num_robots = num_robots
        self.num_states = num_states
        self.num_actions = num_actions
        self.cfg = config or SwarmConfig()
        
        self.lr = self.cfg.learning_rate
        self.gamma = self.cfg.discount_factor
        self.epsilon = self.cfg.epsilon_start
        self.epsilon_min = self.cfg.epsilon_min
        self.epsilon_decay = self.cfg.epsilon_decay
        
        # Q-tables: shape (num_robots, num_states, num_actions)
        self.q_tables = np.zeros((self.num_robots, self.num_states, self.num_actions), dtype=np.float32)

    def select_action(self, agent_id: int, state: int, training: bool = True) -> int:
        """
        Selects an action using epsilon-greedy policy.
        """
        if training and np.random.rand() < self.epsilon:
            return int(np.random.randint(0, self.num_actions))
        
        q_vals = self.q_tables[agent_id, state]
        max_val = np.max(q_vals)
        best_actions = np.where(q_vals == max_val)[0]
        return int(np.random.choice(best_actions))

    def select_actions(self, states: List[int], training: bool = True) -> List[int]:
        """Selects actions for all swarm agents."""
        return [self.select_action(i, states[i], training=training) for i in range(self.num_robots)]

    def update(self, agent_id: int, state: int, action: int, reward: float, next_state: int, next_action: int, done: bool):
        """
        SARSA on-policy TD update rule.
        Uses Q(s', a') from the action actually chosen by the behavior policy.
        """
        current_q = self.q_tables[agent_id, state, action]
        
        if done:
            target = reward
        else:
            # On-policy evaluation: use the actual chosen next action
            target = reward + self.gamma * self.q_tables[agent_id, next_state, next_action]
            
        td_error = target - current_q
        self.q_tables[agent_id, state, action] += self.lr * td_error

    def step_decay(self):
        """Decays epsilon after each episode."""
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
    def get_stats(self) -> dict:
        """Returns summary statistics for logging."""
        return {
            "epsilon": float(self.epsilon),
            "mean_q_val": float(np.mean(self.q_tables)),
            "max_q_val": float(np.max(self.q_tables)),
            "non_zero_q_entries": int(np.count_nonzero(self.q_tables))
        }
