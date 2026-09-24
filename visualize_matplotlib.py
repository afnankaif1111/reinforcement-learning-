"""
Matplotlib Visualizer for Swarm Robotics Mapping and Hazard Detection.
Generates:
1. Multi-panel static visualization of Ground Truth vs Collective Explored Map vs Trajectories.
2. Comparative learning curves across algorithms.
3. Optional step-by-step animation.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Safe headless backend for saving figures
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap

from config import (
    SwarmConfig, CELL_FREE, CELL_OBSTACLE, CELL_HAZARD,
    MAP_UNEXPLORED, MAP_FREE, MAP_OBSTACLE, MAP_HAZARD
)
from environment import SwarmGridEnv
from algorithms import IndependentQLearning, SARSALearning, CooperativeQLearning

# Color scheme
COLOR_FOG = "#1f242d"         # Dark charcoal for unexplored
COLOR_FREE = "#e8f4f8"        # Crisp light cyan for free space
COLOR_OBSTACLE = "#3a4151"    # Slate grey for walls
COLOR_HAZARD = "#ff4757"      # Vibrant red for danger zones
COLOR_HAZARD_FLAG = "#ffa502" # Amber flag for detected hazards
ROBOT_COLORS = ["#2ed573", "#1e90ff", "#9b59b6", "#e67e22", "#00d2d3", "#ff6b81"]

def render_environment_state(env: SwarmGridEnv, title_prefix: str = "", save_path: str = "swarm_state.png"):
    """
    Renders a side-by-side view:
    - Left: Ground Truth World (Obstacles, Hazards, Actual Robot Positions & Trails)
    - Right: Swarm's Collective Explored Map (Fog of War, Discovered Hazards, Discovered Walls)
    - Bottom/Right HUD: Telemetry & Coverage Metrics
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), facecolor="#0f1117")
    fig.suptitle(f"{title_prefix} Swarm Collective Mapping & Hazard Detection", fontsize=15, color="#ffffff", fontweight="bold", y=0.96)
    
    # ------------------ PANEL 1: GROUND TRUTH ------------------
    ax1 = axes[0]
    ax1.set_title("Ground Truth Environment (True World)", color="#a4b0be", fontsize=12, pad=10)
    ax1.set_facecolor("#12161f")
    ax1.set_xlim(-0.5, env.width - 0.5)
    ax1.set_ylim(env.height - 0.5, -0.5) # Invert Y so (0,0) is top-left
    ax1.set_aspect("equal")
    
    # Draw Ground Truth Grid
    for y in range(env.height):
        for x in range(env.width):
            val = env.ground_truth[y, x]
            if val == CELL_OBSTACLE:
                rect = patches.Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor=COLOR_OBSTACLE, edgecolor="#2c3240", linewidth=0.5)
                ax1.add_patch(rect)
            elif val == CELL_HAZARD:
                rect = patches.Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor=COLOR_HAZARD, alpha=0.85, edgecolor="#d63031", linewidth=1.0)
                ax1.add_patch(rect)
                ax1.text(x, y, "☣", color="white", fontsize=10, ha="center", va="center", fontweight="bold")
            else:
                rect = patches.Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor="#1e272e", edgecolor="#2d3436", linewidth=0.2)
                ax1.add_patch(rect)

    # Draw Robot Trajectories & Current Positions on Ground Truth
    for i, traj in enumerate(env.robot_trajectories):
        color = ROBOT_COLORS[i % len(ROBOT_COLORS)]
        if len(traj) > 1:
            xs, ys = zip(*traj)
            ax1.plot(xs, ys, color=color, alpha=0.5, linewidth=1.5, linestyle="--")
            
        cur_x, cur_y = env.robot_positions[i]
        # Sensing footprint circle
        circle = patches.Circle((cur_x, cur_y), env.sensor_range, facecolor=color, alpha=0.15, linestyle=":", edgecolor=color)
        ax1.add_patch(circle)
        # Robot dot
        ax1.scatter(cur_x, cur_y, color=color, s=180, edgecolors="white", linewidth=1.5, zorder=5)
        ax1.text(cur_x, cur_y, str(i + 1), color="white", fontsize=9, ha="center", va="center", fontweight="bold", zorder=6)

    # ------------------ PANEL 2: COLLECTIVE EXPLORED MAP ------------------
    ax2 = axes[1]
    coverage_pct = env.get_coverage_ratio() * 100.0
    hazards_pct = env.get_hazard_detection_rate() * 100.0
    ax2.set_title(f"Collective Swarm Map (Coverage: {coverage_pct:.1f}% | Hazards: {hazards_pct:.1f}%)", color="#a4b0be", fontsize=12, pad=10)
    ax2.set_facecolor("#12161f")
    ax2.set_xlim(-0.5, env.width - 0.5)
    ax2.set_ylim(env.height - 0.5, -0.5)
    ax2.set_aspect("equal")

    for y in range(env.height):
        for x in range(env.width):
            cell = env.collective_map[y, x]
            if cell == MAP_UNEXPLORED:
                # Fog of War
                rect = patches.Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor=COLOR_FOG, edgecolor="#151922", linewidth=0.3)
                ax2.add_patch(rect)
                ax2.text(x, y, "?", color="#34495e", fontsize=7, ha="center", va="center")
            elif cell == MAP_OBSTACLE:
                rect = patches.Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor=COLOR_OBSTACLE, edgecolor="#2c3240", linewidth=0.5)
                ax2.add_patch(rect)
            elif cell == MAP_HAZARD:
                rect = patches.Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor=COLOR_HAZARD_FLAG, alpha=0.9, edgecolor="#ff9f43", linewidth=1.0)
                ax2.add_patch(rect)
                ax2.text(x, y, "⚠️", color="#1e272e", fontsize=9, ha="center", va="center", fontweight="bold")
            else: # MAP_FREE
                rect = patches.Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor=COLOR_FREE, edgecolor="#ced6e0", linewidth=0.3)
                ax2.add_patch(rect)

    # Draw Robots on Explored Map
    for i, (rx, ry) in enumerate(env.robot_positions):
        color = ROBOT_COLORS[i % len(ROBOT_COLORS)]
        ax2.scatter(rx, ry, color=color, s=180, edgecolors="white", linewidth=1.5, zorder=5)
        ax2.text(rx, ry, str(i + 1), color="white", fontsize=9, ha="center", va="center", fontweight="bold", zorder=6)

    # Clean axes ticks
    for ax in [ax1, ax2]:
        ax.set_xticks(range(0, env.width, 5))
        ax.set_yticks(range(0, env.height, 5))
        ax.tick_params(colors="#57606f")
        for spine in ax.spines.values():
            spine.set_color("#2f3542")

    # Add Legend below
    legend_elements = [
        patches.Patch(facecolor=COLOR_FOG, edgecolor="#57606f", label="Fog of War (Unexplored)"),
        patches.Patch(facecolor=COLOR_FREE, label="Explored Free Space"),
        patches.Patch(facecolor=COLOR_OBSTACLE, label="Wall / Obstacle"),
        patches.Patch(facecolor=COLOR_HAZARD, label="True Hazard Zone"),
        patches.Patch(facecolor=COLOR_HAZARD_FLAG, label="Discovered Hazard Alert ⚠️"),
        patches.Patch(facecolor=ROBOT_COLORS[0], label="Swarm Agent")
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=6, frameon=True, facecolor="#1e272e", edgecolor="#2f3542", labelcolor="white", fontsize=9)
    plt.tight_layout(rect=[0, 0.05, 1, 0.94])
    
    plt.savefig(save_path, dpi=180, facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved visualization snapshot to: {save_path}")

def plot_benchmark_curves(json_path: str = "benchmark_results.json", save_path: str = "benchmark_comparison.png"):
    """
    Plots multi-algorithm comparison:
    1. Coverage Rate (%) over training episodes
    2. Hazard Detection Rate (%)
    3. Episodic Reward
    4. Hazard Collisions (Safety Analysis: Q-Learning vs SARSA)
    """
    if not os.path.exists(json_path):
        print(f"No benchmark file found at {json_path}. Run simulation.py --compare first.")
        return

    with open(json_path, "r") as f:
        data = json.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor="#0f1117")
    fig.suptitle("Comparative Evaluation: Swarm Reinforcement Learning Algorithms", fontsize=16, color="#ffffff", fontweight="bold", y=0.98)

    algo_colors = {
        "q_learning": "#ff4757",
        "sarsa": "#2ed573",
        "cooperative": "#1e90ff",
        "dqn": "#ffa502"
    }
    algo_names = {
        "q_learning": "Independent Q-Learning",
        "sarsa": "SARSA (On-Policy)",
        "cooperative": "Cooperative Shared Q",
        "dqn": "Deep Q-Network (DQN)"
    }

    def smooth(vals, window=5):
        if len(vals) < window:
            return vals
        return np.convolve(vals, np.ones(window)/window, mode="valid")

    # Subplot 1: Map Coverage %
    ax_cov = axes[0, 0]
    ax_cov.set_title("1. Map Coverage % vs Training Episodes", color="#e4e7eb", fontsize=12)
    for algo, vals in data.items():
        curve = smooth(vals["coverage_curve"])
        ax_cov.plot(range(1, len(curve) + 1), curve, label=algo_names.get(algo, algo), color=algo_colors.get(algo, "white"), linewidth=2)
    ax_cov.set_xlabel("Episode", color="#a4b0be")
    ax_cov.set_ylabel("Explored Area (%)", color="#a4b0be")
    ax_cov.grid(True, color="#2f3542", linestyle="--", alpha=0.6)
    ax_cov.set_facecolor("#161a23")

    # Subplot 2: Hazard Detection Rate %
    ax_haz = axes[0, 1]
    ax_haz.set_title("2. Hazard Discovery Rate (%)", color="#e4e7eb", fontsize=12)
    for algo, vals in data.items():
        curve = smooth(vals["hazard_curve"])
        ax_haz.plot(range(1, len(curve) + 1), curve, label=algo_names.get(algo, algo), color=algo_colors.get(algo, "white"), linewidth=2)
    ax_haz.set_xlabel("Episode", color="#a4b0be")
    ax_haz.set_ylabel("Hazards Discovered (%)", color="#a4b0be")
    ax_haz.grid(True, color="#2f3542", linestyle="--", alpha=0.6)
    ax_haz.set_facecolor("#161a23")

    # Subplot 3: Episodic Mean Reward
    ax_rew = axes[1, 0]
    ax_rew.set_title("3. Mean Episodic Reward (Convergence)", color="#e4e7eb", fontsize=12)
    for algo, vals in data.items():
        curve = smooth(vals["reward_curve"])
        ax_rew.plot(range(1, len(curve) + 1), curve, label=algo_names.get(algo, algo), color=algo_colors.get(algo, "white"), linewidth=2)
    ax_rew.set_xlabel("Episode", color="#a4b0be")
    ax_rew.set_ylabel("Cumulative Reward", color="#a4b0be")
    ax_rew.grid(True, color="#2f3542", linestyle="--", alpha=0.6)
    ax_rew.set_facecolor("#161a23")

    # Subplot 4: Hazard Collisions / Safety Evaluation
    ax_safe = axes[1, 1]
    ax_safe.set_title("4. Hazard Step Incursions (Safety: Q-Learning vs SARSA)", color="#e4e7eb", fontsize=12)
    for algo, vals in data.items():
        curve = smooth(vals["hazard_hits_curve"])
        ax_safe.plot(range(1, len(curve) + 1), curve, label=algo_names.get(algo, algo), color=algo_colors.get(algo, "white"), linewidth=2)
    ax_safe.set_xlabel("Episode", color="#a4b0be")
    ax_safe.set_ylabel("Hazard Hits (Lower = Safer)", color="#a4b0be")
    ax_safe.grid(True, color="#2f3542", linestyle="--", alpha=0.6)
    ax_safe.set_facecolor("#161a23")

    for row in axes:
        for ax in row:
            ax.tick_params(colors="#a4b0be")
            for spine in ax.spines.values():
                spine.set_color("#2f3542")
            ax.legend(facecolor="#1e272e", edgecolor="#2f3542", labelcolor="white", fontsize=9)

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    plt.savefig(save_path, dpi=180, facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved benchmark comparison figure to: {save_path}")

if __name__ == "__main__":
    # Test rendering a sample environment step
    env = SwarmGridEnv(SwarmConfig(num_robots=4, grid_width=20, grid_height=20))
    # Execute 15 random steps to uncover part of the map
    for _ in range(15):
        random_actions = [np.random.randint(0, 5) for _ in range(4)]
        env.step(random_actions)
    render_environment_state(env, title_prefix="[Demonstration]", save_path="sample_exploration.png")
