"""
Configuration parameters for Swarm Robot Reinforcement Learning Simulation.
"""

from dataclasses import dataclass

@dataclass
class SwarmConfig:
    # Environment Settings
    grid_width: int = 20
    grid_height: int = 20
    obstacle_density: float = 0.12     # Probability of obstacle placement
    num_hazards: int = 8               # Number of danger zones (toxic, fire, structural)
    hazard_radius: int = 1             # Spread radius around hazard center
    
    # Swarm Parameters
    num_robots: int = 4                # Number of collaborative robots
    sensor_range: int = 2              # Sensor radius (Manhattan / Chebyshev)
    max_steps_per_episode: int = 150   # Max steps per exploration episode
    
    # Reinforcement Learning Hyperparameters
    learning_rate: float = 0.15        # Alpha (step size)
    discount_factor: float = 0.95      # Gamma (future reward discount)
    epsilon_start: float = 1.0         # Initial exploration rate
    epsilon_min: float = 0.05          # Minimum exploration rate
    epsilon_decay: float = 0.985       # Decay per episode
    
    # Reward Weights
    reward_new_cell: float = 8.0       # Reward for uncovering an unexplored cell
    reward_hazard_detected: float = 25.0  # Reward for detecting/flagging a hazard
    penalty_hazard_step: float = -20.0 # Penalty for stepping into an active hazard
    penalty_obstacle_collision: float = -5.0 # Penalty for running into wall/obstacle
    penalty_crowding: float = -2.0     # Penalty for stepping directly adjacent to another robot
    cost_step: float = -0.1            # Tiny time penalty to encourage swift mapping
    
    # Training / Experiment Settings
    num_episodes: int = 200            # Total training episodes
    eval_episodes: int = 10            # Evaluation episodes
    seed: int = 42                     # Reproducibility seed

# Cell type constants
CELL_FREE = 0
CELL_OBSTACLE = 1
CELL_HAZARD = 2

# Map discovery states
MAP_UNEXPLORED = -1
MAP_FREE = 0
MAP_OBSTACLE = 1
MAP_HAZARD = 2

# Action definitions
ACTION_UP = 0      # North (-Y)
ACTION_RIGHT = 1   # East (+X)
ACTION_DOWN = 2    # South (+Y)
ACTION_LEFT = 3    # West (-X)
ACTION_STAY = 4    # Stay / Inspect

ACTION_NAMES = ["North", "East", "South", "West", "Stay"]
NUM_ACTIONS = 5
