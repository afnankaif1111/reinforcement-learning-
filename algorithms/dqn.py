"""
Lightweight Deep Q-Network (DQN) for Swarm Robotics.
Provides neural function approximation over the feature state vector.

Architecture:
    Input (11 features) -> Linear(64) -> ReLU -> Linear(32) -> ReLU -> Linear(5 actions)
Features:
    - Experience Replay Buffer
    - Target Q-Network for stable training
    - Epsilon-greedy exploration
"""

import random
from collections import deque
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from config import SwarmConfig, NUM_ACTIONS

class QNetwork(nn.Module):
    def __init__(self, input_dim: int = 11, output_dim: int = NUM_ACTIONS):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class SwarmDQN:
    """
    Lightweight Deep Q-Network for Swarm Robots.
    Uses shared experience replay across the swarm to accelerate sample efficiency.
    """
    def __init__(self, num_robots: int, input_dim: int = 11, num_actions: int = NUM_ACTIONS, config: SwarmConfig = None):
        self.num_robots = num_robots
        self.input_dim = input_dim
        self.num_actions = num_actions
        self.cfg = config or SwarmConfig()
        
        self.lr = 0.001
        self.gamma = self.cfg.discount_factor
        self.epsilon = self.cfg.epsilon_start
        self.epsilon_min = self.cfg.epsilon_min
        self.epsilon_decay = self.cfg.epsilon_decay
        
        self.batch_size = 64
        self.target_update_freq = 5  # episodes
        self.memory = deque(maxlen=10000)
        
        # Policy Network and Target Network
        self.policy_net = QNetwork(input_dim, num_actions)
        self.target_net = QNetwork(input_dim, num_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        
    def state_to_features(self, state_int: int) -> np.ndarray:
        """
        Deconstructs the integer state [0, 319] into an 11-dimensional feature vector:
        - 5-dim one-hot frontier quadrant (N, E, S, W, None)
        - 4-dim binary obstacle flags (N, E, S, W)
        - 1-dim hazard nearby flag
        - 1-dim neighbor crowding flag
        """
        # Deconstruct integer index:
        # state_idx = frontier_dir * 64 + obstacle_bits * 4 + hazard_nearby * 2 + neighbor_nearby
        neighbor_nearby = state_int % 2
        rem1 = state_int // 2
        hazard_nearby = rem1 % 2
        rem2 = rem1 // 2
        obstacle_bits = rem2 % 16
        frontier_dir = rem2 // 16
        
        vec = np.zeros(self.input_dim, dtype=np.float32)
        # Frontier one-hot (first 5 elements)
        if 0 <= frontier_dir < 5:
            vec[frontier_dir] = 1.0
            
        # Obstacle bits (next 4 elements)
        for i in range(4):
            if obstacle_bits & (1 << i):
                vec[5 + i] = 1.0
                
        # Hazard and neighbor flags
        vec[9] = float(hazard_nearby)
        vec[10] = float(neighbor_nearby)
        return vec

    def select_action(self, agent_id: int, state_int: int, training: bool = True) -> int:
        """Selects action via epsilon-greedy using the Q-network."""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
            
        features = self.state_to_features(state_int)
        state_tensor = torch.FloatTensor(features).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
            return int(q_values.argmax(dim=1).item())

    def select_actions(self, states: List[int], training: bool = True) -> List[int]:
        """Selects actions for all swarm agents."""
        return [self.select_action(i, states[i], training=training) for i in range(self.num_robots)]

    def remember(self, state: int, action: int, reward: float, next_state: int, done: bool):
        """Stores transition in replay buffer."""
        feat_s = self.state_to_features(state)
        feat_next_s = self.state_to_features(next_state)
        self.memory.append((feat_s, action, reward, feat_next_s, done))

    def update(self):
        """Samples a minibatch and performs gradient descent step."""
        if len(self.memory) < self.batch_size:
            return
            
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states_t = torch.FloatTensor(np.array(states))
        actions_t = torch.LongTensor(actions).unsqueeze(1)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1)
        next_states_t = torch.FloatTensor(np.array(next_states))
        dones_t = torch.FloatTensor(dones).unsqueeze(1)
        
        # Current Q estimates: Q(s, a)
        curr_q = self.policy_net(states_t).gather(1, actions_t)
        
        # Target Q calculation: r + gamma * max_a' Q_target(s', a')
        with torch.no_grad():
            max_next_q = self.target_net(next_states_t).max(1)[0].unsqueeze(1)
            target_q = rewards_t + (1.0 - dones_t) * self.gamma * max_next_q
            
        loss = self.criterion(curr_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        """Copies policy network weights to target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def step_decay(self, episode: int):
        """Decays epsilon and periodically synchronizes target network."""
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        if episode % self.target_update_freq == 0:
            self.update_target_network()

    def get_stats(self) -> dict:
        return {
            "epsilon": float(self.epsilon),
            "replay_size": len(self.memory)
        }
