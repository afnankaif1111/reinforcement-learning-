"""
Swarm Robotics Reinforcement Learning Simulator.
Handles training loops, evaluation benchmarks, and comparative metric tracking across:
- Independent Q-Learning (IQL)
- SARSA
- Cooperative Shared-Map Q-Learning
- Deep Q-Network (DQN)
"""

import os
import sys
import json
import argparse
import time
from typing import Dict, List, Any
import numpy as np

from config import SwarmConfig
from environment import SwarmGridEnv
from algorithms import IndependentQLearning, SARSALearning, CooperativeQLearning, SwarmDQN

class SwarmSimulation:
    def __init__(self, algorithm_name: str = "q_learning", config: SwarmConfig = None):
        self.cfg = config or SwarmConfig()
        self.algo_name = algorithm_name.lower()
        self.env = SwarmGridEnv(self.cfg)
        
        # Instantiate selected algorithm
        if self.algo_name == "q_learning":
            self.model = IndependentQLearning(
                num_robots=self.cfg.num_robots,
                num_states=320,
                num_actions=5,
                config=self.cfg
            )
        elif self.algo_name == "sarsa":
            self.model = SARSALearning(
                num_robots=self.cfg.num_robots,
                num_states=320,
                num_actions=5,
                config=self.cfg
            )
        elif self.algo_name == "cooperative":
            self.model = CooperativeQLearning(
                num_robots=self.cfg.num_robots,
                num_states=320,
                num_actions=5,
                config=self.cfg
            )
        elif self.algo_name == "dqn":
            self.model = SwarmDQN(
                num_robots=self.cfg.num_robots,
                input_dim=11,
                num_actions=5,
                config=self.cfg
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.algo_name}. Choose from 'q_learning', 'sarsa', 'cooperative', 'dqn'.")
            
        self.history: Dict[str, List[Any]] = {
            "episode": [],
            "mean_reward": [],
            "coverage_pct": [],
            "hazards_detected_pct": [],
            "total_collisions": [],
            "total_hazard_hits": [],
            "steps": [],
            "epsilon": []
        }

    def run_episode(self, episode_num: int, training: bool = True) -> Dict[str, float]:
        """Runs a single episode of mapping and hazard search."""
        states = self.env.reset()
        done = False
        
        ep_rewards = [0.0] * self.cfg.num_robots
        ep_collisions = 0
        ep_hazard_hits = 0
        step_count = 0
        
        # For SARSA, we need the initial action selected upfront
        if self.algo_name == "sarsa":
            actions = self.model.select_actions(states, training=training)
            
        while not done:
            step_count += 1
            
            if self.algo_name != "sarsa":
                actions = self.model.select_actions(states, training=training)
                
            next_states, rewards, done, info = self.env.step(actions)
            
            ep_collisions += info["step_collisions"]
            ep_hazard_hits += info["step_hazard_hits"]
            for i in range(self.cfg.num_robots):
                ep_rewards[i] += rewards[i]
                
            # Algorithm update steps
            if training:
                if self.algo_name == "q_learning":
                    for i in range(self.cfg.num_robots):
                        self.model.update(i, states[i], actions[i], rewards[i], next_states[i], done)
                        
                elif self.algo_name == "sarsa":
                    next_actions = self.model.select_actions(next_states, training=training)
                    for i in range(self.cfg.num_robots):
                        self.model.update(i, states[i], actions[i], rewards[i], next_states[i], next_actions[i], done)
                    actions = next_actions  # for next iteration
                    
                elif self.algo_name == "cooperative":
                    self.model.update_team(states, actions, rewards, next_states, done)
                    
                elif self.algo_name == "dqn":
                    for i in range(self.cfg.num_robots):
                        self.model.remember(states[i], actions[i], rewards[i], next_states[i], done)
                    self.model.update()
                    
            states = next_states

        # End of episode decay
        if training:
            if self.algo_name == "dqn":
                self.model.step_decay(episode_num)
            else:
                self.model.step_decay()

        coverage = self.env.get_coverage_ratio() * 100.0
        hazard_rate = self.env.get_hazard_detection_rate() * 100.0
        mean_rew = float(np.mean(ep_rewards))
        
        return {
            "episode": episode_num,
            "mean_reward": mean_rew,
            "coverage_pct": coverage,
            "hazards_detected_pct": hazard_rate,
            "total_collisions": ep_collisions,
            "total_hazard_hits": ep_hazard_hits,
            "steps": step_count,
            "epsilon": getattr(self.model, "epsilon", 0.0)
        }

    def train(self, num_episodes: int = None, verbose: bool = True) -> Dict[str, List[Any]]:
        """Trains the swarm for a specified number of episodes."""
        total_eps = num_episodes or self.cfg.num_episodes
        start_time = time.time()
        
        if verbose:
            print(f"\n========================================================")
            print(f" Starting Swarm Training: {self.algo_name.upper()}")
            print(f" Robots: {self.cfg.num_robots} | Grid: {self.cfg.grid_width}x{self.cfg.grid_height} | Episodes: {total_eps}")
            print(f"========================================================")
            
        for ep in range(1, total_eps + 1):
            ep_metrics = self.run_episode(ep, training=True)
            for k, v in ep_metrics.items():
                self.history[k].append(v)
                
            if verbose and (ep % 20 == 0 or ep == 1 or ep == total_eps):
                print(f"[Ep {ep:3d}/{total_eps}] "
                      f"Coverage: {ep_metrics['coverage_pct']:5.1f}% | "
                      f"Hazards: {ep_metrics['hazards_detected_pct']:5.1f}% | "
                      f"Reward: {ep_metrics['mean_reward']:7.1f} | "
                      f"HazHits: {ep_metrics['total_hazard_hits']:2d} | "
                      f"Eps: {ep_metrics['epsilon']:.3f}")
                      
        elapsed = time.time() - start_time
        if verbose:
            print(f"Training completed in {elapsed:.2f} seconds.")
            
        return self.history

    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """Evaluates trained policy with epsilon=0 (pure exploitation)."""
        metrics_list = []
        for ep in range(num_episodes):
            m = self.run_episode(ep, training=False)
            metrics_list.append(m)
            
        avg_metrics = {
            "eval_coverage_pct": float(np.mean([m["coverage_pct"] for m in metrics_list])),
            "eval_hazard_rate_pct": float(np.mean([m["hazards_detected_pct"] for m in metrics_list])),
            "eval_mean_reward": float(np.mean([m["mean_reward"] for m in metrics_list])),
            "eval_hazard_hits": float(np.mean([m["total_hazard_hits"] for m in metrics_list])),
            "eval_collisions": float(np.mean([m["total_collisions"] for m in metrics_list])),
            "eval_steps": float(np.mean([m["steps"] for m in metrics_list]))
        }
        return avg_metrics

def compare_all_algorithms(episodes: int = 120) -> Dict[str, Dict[str, Any]]:
    """Runs a side-by-side benchmark comparing all four algorithms."""
    algorithms = ["q_learning", "sarsa", "cooperative", "dqn"]
    results = {}
    
    print("\n" + "#" * 65)
    print("  SWARM ROBOTICS RL BENCHMARK COMPARISON")
    print("#" * 65)
    
    for algo in algorithms:
        sim = SwarmSimulation(algorithm_name=algo)
        history = sim.train(num_episodes=episodes, verbose=True)
        eval_stats = sim.evaluate(num_episodes=10)
        results[algo] = {
            "history": history,
            "eval_summary": eval_stats
        }
        
    print("\n" + "=" * 80)
    print(f"{'Algorithm':<15} | {'Coverage %':<12} | {'Hazards Found %':<16} | {'Hazard Hits':<12} | {'Reward':<10}")
    print("=" * 80)
    for algo, res in results.items():
        ev = res["eval_summary"]
        print(f"{algo:<15} | {ev['eval_coverage_pct']:<12.1f} | {ev['eval_hazard_rate_pct']:<16.1f} | {ev['eval_hazard_hits']:<12.2f} | {ev['eval_mean_reward']:<10.1f}")
    print("=" * 80)
    
    # Save results to JSON
    with open("benchmark_results.json", "w") as f:
        # Convert history arrays to serializable lists
        json_ready = {}
        for algo, d in results.items():
            json_ready[algo] = {
                "eval_summary": d["eval_summary"],
                "coverage_curve": [float(x) for x in d["history"]["coverage_pct"]],
                "hazard_curve": [float(x) for x in d["history"]["hazards_detected_pct"]],
                "reward_curve": [float(x) for x in d["history"]["mean_reward"]],
                "hazard_hits_curve": [int(x) for x in d["history"]["total_hazard_hits"]]
            }
        json.dump(json_ready, f, indent=2)
    print("Benchmark results saved to 'benchmark_results.json'.")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Swarm RL Simulation Runner")
    parser.add_argument("--algorithm", type=str, default="cooperative",
                        choices=["q_learning", "sarsa", "cooperative", "dqn", "all"],
                        help="RL Algorithm to train")
    parser.add_argument("--episodes", type=int, default=100, help="Number of training episodes")
    parser.add_argument("--robots", type=int, default=4, help="Number of robots in the swarm")
    parser.add_argument("--compare", action="store_true", help="Run full benchmark comparing all 4 algorithms")
    
    args = parser.parse_args()
    
    cfg = SwarmConfig(num_robots=args.robots, num_episodes=args.episodes)
    
    if args.compare or args.algorithm == "all":
        compare_all_algorithms(episodes=args.episodes)
    else:
        sim = SwarmSimulation(algorithm_name=args.algorithm, config=cfg)
        sim.train(num_episodes=args.episodes, verbose=True)
        eval_res = sim.evaluate(num_episodes=10)
        print("\n--- Final Evaluation (10 Test Episodes) ---")
        for k, v in eval_res.items():
            print(f"  {k}: {v:.2f}")
