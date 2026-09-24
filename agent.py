"""
Swarm Robot Agent representation.
Tracks individual telemetry, path history, and localized hazard alerts.
"""

from typing import Tuple, List

class SwarmRobot:
    """
    Represents an individual physical agent within the swarm.
    """
    def __init__(self, robot_id: int, initial_pos: Tuple[int, int], sensor_range: int = 2):
        self.robot_id = robot_id
        self.x, self.y = initial_pos
        self.sensor_range = sensor_range
        
        # Telemetry
        self.steps_taken = 0
        self.collisions = 0
        self.hazard_hits = 0
        self.cells_explored = 0
        self.hazards_discovered = 0
        self.total_reward = 0.0
        self.path_history: List[Tuple[int, int]] = [initial_pos]
        
    def move_to(self, new_pos: Tuple[int, int]):
        """Updates robot position and records trajectory."""
        self.x, self.y = new_pos
        self.steps_taken += 1
        self.path_history.append(new_pos)

    def reset(self, start_pos: Tuple[int, int]):
        """Resets the robot for a new episode."""
        self.x, self.y = start_pos
        self.steps_taken = 0
        self.collisions = 0
        self.hazard_hits = 0
        self.cells_explored = 0
        self.hazards_discovered = 0
        self.total_reward = 0.0
        self.path_history = [start_pos]

    @property
    def position(self) -> Tuple[int, int]:
        return (self.x, self.y)
