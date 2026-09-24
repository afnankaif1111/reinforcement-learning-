"""
Swarm Environment: 2D GridWorld with Obstacles, Hazards, and Occupancy Grid.
Supports decentralized partial observability, collective mapping, and hazard detection.
"""

import numpy as np
from typing import List, Tuple, Dict, Any
from config import (
    SwarmConfig, CELL_FREE, CELL_OBSTACLE, CELL_HAZARD,
    MAP_UNEXPLORED, MAP_FREE, MAP_OBSTACLE, MAP_HAZARD,
    ACTION_UP, ACTION_RIGHT, ACTION_DOWN, ACTION_LEFT, ACTION_STAY
)

class SwarmGridEnv:
    """
    Simulates a 2D unknown search area for a swarm of robots.
    - True world has obstacles and hazards.
    - Swarm builds a collective (or individual) occupancy grid.
    - Handles line-of-sight sensor updates, collisions, and rewards.
    """
    def __init__(self, config: SwarmConfig = None):
        self.cfg = config or SwarmConfig()
        self.width = self.cfg.grid_width
        self.height = self.cfg.grid_height
        self.num_robots = self.cfg.num_robots
        self.sensor_range = self.cfg.sensor_range
        
        # State containers
        self.ground_truth = np.zeros((self.height, self.width), dtype=np.int32)
        self.collective_map = np.full((self.height, self.width), MAP_UNEXPLORED, dtype=np.int32)
        self.individual_maps = [
            np.full((self.height, self.width), MAP_UNEXPLORED, dtype=np.int32)
            for _ in range(self.num_robots)
        ]
        
        self.robot_positions: List[Tuple[int, int]] = []
        self.robot_trajectories: List[List[Tuple[int, int]]] = [[] for _ in range(self.num_robots)]
        
        self.hazard_locations: List[Tuple[int, int]] = []
        self.discovered_hazards = set()
        
        self.current_step = 0
        self.navigable_cells_count = 0
        self.total_hazards_count = 0
        
        self.rng = np.random.RandomState(self.cfg.seed)
        self.reset()

    def reset(self, seed: int = None) -> List[int]:
        """Resets the environment, generates a fresh map, and places robots."""
        if seed is not None:
            self.rng = np.random.RandomState(seed)
            
        self.current_step = 0
        self.ground_truth.fill(CELL_FREE)
        self.collective_map.fill(MAP_UNEXPLORED)
        for m in self.individual_maps:
            m.fill(MAP_UNEXPLORED)
            
        self.discovered_hazards.clear()
        self.robot_trajectories = [[] for _ in range(self.num_robots)]
        
        # 1. Place boundary walls
        self.ground_truth[0, :] = CELL_OBSTACLE
        self.ground_truth[-1, :] = CELL_OBSTACLE
        self.ground_truth[:, 0] = CELL_OBSTACLE
        self.ground_truth[:, -1] = CELL_OBSTACLE
        
        # 2. Place random obstacles
        num_obstacles = int(self.cfg.obstacle_density * (self.width - 2) * (self.height - 2))
        for _ in range(num_obstacles):
            x = self.rng.randint(1, self.width - 1)
            y = self.rng.randint(1, self.height - 1)
            self.ground_truth[y, x] = CELL_OBSTACLE
            
        # 3. Place hazards (danger zones)
        self.hazard_locations.clear()
        placed_hazards = 0
        attempts = 0
        while placed_hazards < self.cfg.num_hazards and attempts < 200:
            attempts += 1
            hx = self.rng.randint(2, self.width - 2)
            hy = self.rng.randint(2, self.height - 2)
            if self.ground_truth[hy, hx] == CELL_FREE:
                self.ground_truth[hy, hx] = CELL_HAZARD
                self.hazard_locations.append((hx, hy))
                placed_hazards += 1
                
        self.total_hazards_count = len(self.hazard_locations)
        self.navigable_cells_count = int(np.sum(self.ground_truth != CELL_OBSTACLE))
        
        # 4. Spawn swarm robots at safe start positions (typically clustered near an entrance)
        self.robot_positions.clear()
        start_attempts = 0
        while len(self.robot_positions) < self.num_robots and start_attempts < 500:
            start_attempts += 1
            # Cluster near top-left or center
            sx = self.rng.randint(1, max(3, self.width // 4))
            sy = self.rng.randint(1, max(3, self.height // 4))
            if self.ground_truth[sy, sx] == CELL_FREE and (sx, sy) not in self.robot_positions:
                self.robot_positions.append((sx, sy))
                
        # Fallback if crowded
        if len(self.robot_positions) < self.num_robots:
            for y in range(1, self.height - 1):
                for x in range(1, self.width - 1):
                    if self.ground_truth[y, x] == CELL_FREE and (x, y) not in self.robot_positions:
                        self.robot_positions.append((x, y))
                        if len(self.robot_positions) == self.num_robots:
                            break
                if len(self.robot_positions) == self.num_robots:
                    break

        for i, pos in enumerate(self.robot_positions):
            self.robot_trajectories[i].append(pos)
            
        # 5. Perform initial sensor sweep
        self._update_sensors_and_mapping()
        
        # Return initial compact state representation for each robot
        return [self.get_compact_state(i) for i in range(self.num_robots)]

    def _update_sensors_and_mapping(self) -> Tuple[List[int], List[int]]:
        """
        Updates the collective and individual occupancy maps based on each robot's sensor sweep.
        Returns:
            new_cells_per_robot: list of newly uncovered cells count for each robot
            new_hazards_per_robot: list of newly discovered hazards count for each robot
        """
        new_cells_per_robot = [0] * self.num_robots
        new_hazards_per_robot = [0] * self.num_robots
        
        for i, (rx, ry) in enumerate(self.robot_positions):
            # Scan in square bounding box [-sensor_range, +sensor_range]
            for dy in range(-self.sensor_range, self.sensor_range + 1):
                for dx in range(-self.sensor_range, self.sensor_range + 1):
                    nx, ny = rx + dx, ry + dy
                    if 0 <= nx < self.width and 0 <= ny < self.height:
                        # Check Euclidean distance for circular sensing radius
                        if dx*dx + dy*dy <= self.sensor_range * self.sensor_range + 0.5:
                            true_val = self.ground_truth[ny, nx]
                            
                            # Update individual map
                            if self.individual_maps[i][ny, nx] == MAP_UNEXPLORED:
                                self.individual_maps[i][ny, nx] = true_val
                                
                            # Update collective shared map
                            if self.collective_map[ny, nx] == MAP_UNEXPLORED:
                                self.collective_map[ny, nx] = true_val
                                new_cells_per_robot[i] += 1
                                
                                # Check if hazard detected
                                if true_val == CELL_HAZARD:
                                    if (nx, ny) not in self.discovered_hazards:
                                        self.discovered_hazards.add((nx, ny))
                                        new_hazards_per_robot[i] += 1
                                        
        return new_cells_per_robot, new_hazards_per_robot

    def step(self, actions: List[int]) -> Tuple[List[int], List[float], bool, Dict[str, Any]]:
        """
        Advances the environment by one step for the swarm.
        actions: list of discrete actions for each robot.
        Returns:
            next_states: List of compact discrete states
            rewards: List of scalar rewards per agent
            done: Whether the episode has terminated
            info: Diagnostics dict (coverage %, hazards found %, collisions)
        """
        self.current_step += 1
        rewards = [self.cfg.cost_step] * self.num_robots
        step_collisions = 0
        step_hazard_hits = 0
        
        new_positions = list(self.robot_positions)
        
        # 1. Execute movement actions
        for i, act in enumerate(actions):
            x, y = self.robot_positions[i]
            target_x, target_y = x, y
            
            if act == ACTION_UP:
                target_y -= 1
            elif act == ACTION_RIGHT:
                target_x += 1
            elif act == ACTION_DOWN:
                target_y += 1
            elif act == ACTION_LEFT:
                target_x -= 1
            elif act == ACTION_STAY:
                pass  # Stationary scan
                
            # Boundary & Obstacle collision check
            if (target_x < 0 or target_x >= self.width or
                target_y < 0 or target_y >= self.height or
                self.ground_truth[target_y, target_x] == CELL_OBSTACLE):
                # Collision: agent stays at current position
                rewards[i] += self.cfg.penalty_obstacle_collision
                step_collisions += 1
                new_positions[i] = (x, y)
            else:
                # Valid move
                new_positions[i] = (target_x, target_y)
                
                # Check if stepped onto a hazard
                if self.ground_truth[target_y, target_x] == CELL_HAZARD:
                    rewards[i] += self.cfg.penalty_hazard_step
                    step_hazard_hits += 1

        self.robot_positions = new_positions
        for i, pos in enumerate(self.robot_positions):
            self.robot_trajectories[i].append(pos)
            
        # 2. Check swarm crowding (penalize if robots crowd the exact same or adjacent cell)
        for i in range(self.num_robots):
            for j in range(i + 1, self.num_robots):
                dist = abs(self.robot_positions[i][0] - self.robot_positions[j][0]) + \
                       abs(self.robot_positions[i][1] - self.robot_positions[j][1])
                if dist <= 1:
                    rewards[i] += self.cfg.penalty_crowding
                    rewards[j] += self.cfg.penalty_crowding

        # 3. Update sensors, reveal map, discover hazards
        new_cells, new_hazards = self._update_sensors_and_mapping()
        for i in range(self.num_robots):
            rewards[i] += new_cells[i] * self.cfg.reward_new_cell
            rewards[i] += new_hazards[i] * self.cfg.reward_hazard_detected

        # 4. Termination condition
        coverage_ratio = self.get_coverage_ratio()
        hazard_detection_ratio = self.get_hazard_detection_rate()
        
        all_explored = coverage_ratio >= 0.98
        max_steps_reached = self.current_step >= self.cfg.max_steps_per_episode
        done = all_explored or max_steps_reached
        
        next_states = [self.get_compact_state(i) for i in range(self.num_robots)]
        
        info = {
            "step": self.current_step,
            "coverage_ratio": coverage_ratio,
            "hazard_detection_rate": hazard_detection_ratio,
            "hazards_found": len(self.discovered_hazards),
            "total_hazards": self.total_hazards_count,
            "step_collisions": step_collisions,
            "step_hazard_hits": step_hazard_hits,
            "all_explored": all_explored
        }
        
        return next_states, rewards, done, info

    def get_compact_state(self, agent_id: int) -> int:
        """
        Converts the agent's local observation into an integer state index in [0, 319].
        
        Components:
        1. frontier_dir in [0..4]: Direction of highest unexplored concentration
           0: North, 1: East, 2: South, 3: West, 4: None
        2. obstacle_bits in [0..15]: 4-bit binary indicating obstacles at [N, E, S, W]
        3. hazard_nearby in [0, 1]: Hazard detected within sensor range
        4. neighbor_nearby in [0, 1]: Another robot within Manhattan distance <= 2
        
        Total states = 5 * 16 * 2 * 2 = 320.
        """
        rx, ry = self.robot_positions[agent_id]
        
        # 1. Frontier direction calculation
        # Count unexplored cells in 4 directional quadrants within a search radius
        search_radius = self.sensor_range * 3
        unexplored_counts = [0, 0, 0, 0] # N, E, S, W
        
        for dy in range(-search_radius, search_radius + 1):
            for dx in range(-search_radius, search_radius + 1):
                qx, qy = rx + dx, ry + dy
                if 0 <= qx < self.width and 0 <= qy < self.height:
                    # Check collective map for unexplored cells
                    if self.collective_map[qy, qx] == MAP_UNEXPLORED:
                        if dy < 0 and abs(dy) >= abs(dx):
                            unexplored_counts[0] += 1 # North
                        elif dx > 0 and abs(dx) >= abs(dy):
                            unexplored_counts[1] += 1 # East
                        elif dy > 0 and abs(dy) >= abs(dx):
                            unexplored_counts[2] += 1 # South
                        elif dx < 0 and abs(dx) >= abs(dy):
                            unexplored_counts[3] += 1 # West
                            
        max_unexplored = max(unexplored_counts)
        if max_unexplored == 0:
            frontier_dir = 4 # None / all local area mapped
        else:
            frontier_dir = int(np.argmax(unexplored_counts))
            
        # 2. Obstacle adjacency bits (North, East, South, West)
        # Bit 0: North, Bit 1: East, Bit 2: South, Bit 3: West
        obstacle_bits = 0
        adjacents = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        for bit_idx, (adx, ady) in enumerate(adjacents):
            ax, ay = rx + adx, ry + ady
            if ax < 0 or ax >= self.width or ay < 0 or ay >= self.height or self.ground_truth[ay, ax] == CELL_OBSTACLE:
                obstacle_bits |= (1 << bit_idx)
                
        # 3. Hazard nearby flag (within sensor range)
        hazard_nearby = 0
        for dy in range(-self.sensor_range, self.sensor_range + 1):
            for dx in range(-self.sensor_range, self.sensor_range + 1):
                hx, hy = rx + dx, ry + dy
                if 0 <= hx < self.width and 0 <= hy < self.height:
                    if self.ground_truth[hy, hx] == CELL_HAZARD:
                        hazard_nearby = 1
                        break
            if hazard_nearby:
                break
                
        # 4. Neighbor nearby flag (Manhattan distance <= 2)
        neighbor_nearby = 0
        for j, (ox, oy) in enumerate(self.robot_positions):
            if j != agent_id:
                if abs(rx - ox) + abs(ry - oy) <= 2:
                    neighbor_nearby = 1
                    break
                    
        # Encode uniquely into integer [0, 319]
        # frontier_dir: 0..4 (5 values)
        # obstacle_bits: 0..15 (16 values)
        # hazard_nearby: 0..1 (2 values)
        # neighbor_nearby: 0..1 (2 values)
        state_idx = (
            frontier_dir * (16 * 2 * 2) +
            obstacle_bits * (2 * 2) +
            hazard_nearby * 2 +
            neighbor_nearby
        )
        return int(state_idx)

    def get_coverage_ratio(self) -> float:
        """Percentage of navigable (non-obstacle) world revealed."""
        revealed_navigable = np.sum((self.collective_map != MAP_UNEXPLORED) & (self.ground_truth != CELL_OBSTACLE))
        return float(revealed_navigable / max(1, self.navigable_cells_count))

    def get_hazard_detection_rate(self) -> float:
        """Percentage of ground truth hazards successfully detected."""
        if self.total_hazards_count == 0:
            return 1.0
        return float(len(self.discovered_hazards) / self.total_hazards_count)
