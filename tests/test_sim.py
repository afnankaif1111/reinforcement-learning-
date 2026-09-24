"""
Unit and Integration Tests for Swarm Robotics RL Simulation.
Verifies environment dynamics, compact state encoding, and all 4 RL algorithms.
"""

import unittest
import numpy as np
import torch

from config import SwarmConfig, CELL_OBSTACLE, CELL_HAZARD, MAP_UNEXPLORED
from environment import SwarmGridEnv
from algorithms import IndependentQLearning, SARSALearning, CooperativeQLearning, SwarmDQN

class TestSwarmEnvironment(unittest.TestCase):
    def setUp(self):
        self.cfg = SwarmConfig(grid_width=15, grid_height=15, num_robots=3, num_hazards=4, seed=123)
        self.env = SwarmGridEnv(self.cfg)

    def test_env_initialization(self):
        self.assertEqual(len(self.env.robot_positions), 3)
        self.assertEqual(self.env.ground_truth.shape, (15, 15))
        # Ensure boundary walls exist
        self.assertTrue(np.all(self.env.ground_truth[0, :] == CELL_OBSTACLE))
        self.assertTrue(np.all(self.env.ground_truth[-1, :] == CELL_OBSTACLE))
        self.assertTrue(np.all(self.env.ground_truth[:, 0] == CELL_OBSTACLE))
        self.assertTrue(np.all(self.env.ground_truth[:, -1] == CELL_OBSTACLE))

    def test_state_encoder_bounds(self):
        for i in range(self.env.num_robots):
            state_idx = self.env.get_compact_state(i)
            self.assertGreaterEqual(state_idx, 0)
            self.assertLess(state_idx, 320, "State index must be strictly in [0, 319]")

    def test_step_execution(self):
        actions = [1, 2, 0]  # East, South, North
        next_states, rewards, done, info = self.env.step(actions)
        self.assertEqual(len(next_states), 3)
        self.assertEqual(len(rewards), 3)
        self.assertIsInstance(done, bool)
        self.assertIn("coverage_ratio", info)
        self.assertIn("hazard_detection_rate", info)
        self.assertGreater(info["coverage_ratio"], 0.0)

class TestRLAlgorithms(unittest.TestCase):
    def setUp(self):
        self.cfg = SwarmConfig(num_robots=2, seed=42)
        self.env = SwarmGridEnv(self.cfg)

    def test_q_learning_update(self):
        agent = IndependentQLearning(num_robots=2, num_states=320, num_actions=5, config=self.cfg)
        states = self.env.reset()
        actions = agent.select_actions(states, training=True)
        next_states, rewards, done, _ = self.env.step(actions)
        
        # Test update
        initial_q = agent.q_tables[0, states[0], actions[0]]
        agent.update(0, states[0], actions[0], 10.0, next_states[0], done)
        updated_q = agent.q_tables[0, states[0], actions[0]]
        self.assertNotEqual(initial_q, updated_q)

    def test_sarsa_update(self):
        agent = SARSALearning(num_robots=2, num_states=320, num_actions=5, config=self.cfg)
        states = self.env.reset()
        actions = agent.select_actions(states, training=True)
        next_states, rewards, done, _ = self.env.step(actions)
        next_actions = agent.select_actions(next_states, training=True)
        
        agent.update(0, states[0], actions[0], -5.0, next_states[0], next_actions[0], done)
        self.assertLess(agent.q_tables[0, states[0], actions[0]], 0.0)

    def test_cooperative_q_update(self):
        agent = CooperativeQLearning(num_robots=2, num_states=320, num_actions=5, config=self.cfg)
        states = self.env.reset()
        actions = agent.select_actions(states, training=True)
        next_states, rewards, done, _ = self.env.step(actions)
        
        agent.update_team(states, actions, rewards, next_states, done)
        self.assertGreater(np.count_nonzero(agent.shared_q_table), 0)

    def test_dqn_forward_backward(self):
        agent = SwarmDQN(num_robots=2, input_dim=11, num_actions=5, config=self.cfg)
        states = self.env.reset()
        actions = agent.select_actions(states, training=True)
        self.assertEqual(len(actions), 2)
        
        # Test feature mapping
        feat = agent.state_to_features(150)
        self.assertEqual(feat.shape, (11,))
        
        # Replay and batch update
        for _ in range(70):
            agent.remember(state=10, action=1, reward=5.0, next_state=12, done=False)
        agent.update()
        self.assertEqual(len(agent.memory), 70)

if __name__ == "__main__":
    unittest.main()
