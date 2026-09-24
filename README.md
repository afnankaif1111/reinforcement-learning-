# Swarm Robotics Reinforcement Learning: Collective Mapping & Hazard Detection

An accessible, modular, and mathematically sound Reinforcement Learning project designed for academic study and research. Multiple autonomous robots explore an unknown grid-world environment under **fog of war**, collaboratively constructing an occupancy grid map while identifying and avoiding localized environmental hazards (e.g., toxic spills, fire zones, or radiation hotspots).

---

## 🚀 Key Highlights & Highlights

1. **Foundational & Intuitive RL Formulations**:
   - Built around the **Decentralized Partially Observable Markov Decision Process (Dec-POMDP)**.
   - Features a **compact 320-state discretization** that prevents the *curse of dimensionality* while preserving interpretable state features:
     - **Frontier Quadrant** (direction of highest unexplored terrain)
     - **Obstacle Proximity** (4-bit binary wall detection)
     - **Hazard Presence** (local danger signature)
     - **Swarm Crowding** (mutual interference avoidance)

2. **Implemented RL Algorithms**:
   - 🟥 **Independent Q-Learning (IQL)**: Classic off-policy TD(0) learning. Aggressive exploration toward unknown frontiers.
   - 🟩 **SARSA (State-Action-Reward-State-Action)**: On-policy TD(0) learning. Demonstrates risk-sensitive safe navigation by learning a safety buffer around hazards.
   - 🟦 **Cooperative Shared-Map Q-Learning**: Multi-agent coordinated learning with team reward blending and shared occupancy updates.
   - 🟧 **Lightweight Deep Q-Network (DQN)**: 2-layer PyTorch neural value approximator with replay buffer and target network.

3. **Dual Visualization Suite**:
   - 🌐 **Interactive Browser GUI (`web_viewer.html`)**: Real-time 60 FPS HTML5 canvas simulation with live HUD, adjustable speed, algorithm switching, and fog-of-war toggles.
   - 📊 **Matplotlib Multi-Panel Dashboards (`visualize_matplotlib.py`)**: Publication-ready comparison charts and ground-truth vs perceived occupancy maps.

---

## 📁 Repository Structure

```
.
├── SWARM_RL_PROJECT_DOCUMENT.md    # Task 1: Complete theoretical course document & formulation
├── README.md                       # Project overview & quickstart guide
├── config.py                       # Hyperparameters, grid settings, and reward weights
├── environment.py                  # 2D GridWorld with obstacles, hazards, and occupancy grid
├── agent.py                        # SwarmRobot agent class and path telemetry
├── algorithms/
│   ├── __init__.py
│   ├── q_learning.py               # Tabular Independent Q-Learning (Off-policy)
│   ├── sarsa.py                    # Tabular SARSA (On-policy, risk-sensitive)
│   ├── cooperative_q.py            # Shared-Map Cooperative Q-Learning
│   └── dqn.py                      # Lightweight PyTorch Deep Q-Network
├── simulation.py                   # Training loop, evaluation engine & comparative benchmark
├── visualize_matplotlib.py         # Matplotlib rendering & curve plotting
├── web_viewer.html                 # Interactive in-browser swarm simulator
├── run_web_gui.py                  # One-click launcher for the browser simulation
├── tests/
│   └── test_sim.py                 # Automated unit and integration test suite
├── benchmark_results.json          # Empirical evaluation results across all 4 algorithms
├── sample_exploration.png          # Side-by-side ground truth vs explored map snapshot
└── benchmark_comparison.png        # Comparative performance curves
```

---

## ⚡ Quickstart Guide

### 1. Launch the Interactive Browser Simulator
You can directly open `web_viewer.html` in Chrome, Safari, or Firefox:
```bash
python3 run_web_gui.py
```
Or open the file directly:
```bash
open web_viewer.html
```
- Click **"▶ Start Swarm"** to watch the robots clear the fog of war in real time.
- Switch between **Cooperative Q**, **SARSA**, **Q-Learning**, and **Random Walk** from the dropdown menu to visually see behavioral differences!

### 2. Run Training & Benchmarking CLI
Train a specific algorithm (e.g. Cooperative Q):
```bash
python3 simulation.py --algorithm cooperative --episodes 100 --robots 4
```

Run a full side-by-side comparison across all 4 algorithms:
```bash
python3 simulation.py --compare --episodes 80
```

### 3. Generate High-Resolution Visualizations
Generate a sample exploration snapshot:
```bash
python3 visualize_matplotlib.py
```

Plot the comparative learning curves from the benchmark:
```bash
python3 -c "import visualize_matplotlib; visualize_matplotlib.plot_benchmark_curves()"
```

### 4. Run Unit Tests
Verify environment integrity, state encoding, and algorithm updates:
```bash
python3 -m unittest discover -s tests
```

---

## 📊 Summary of Experimental Results

In our 80-episode benchmark on a $20 \times 20$ grid with 8 hazards and 4 robots:

| Metric | Q-Learning (Off-Policy) | SARSA (On-Policy) | Cooperative Q | Deep Q-Network (DQN) |
| :--- | :---: | :---: | :---: | :---: |
| **Exploration Strategy** | Aggressive | Conservative | Coordinated | Value approximated |
| **Hazard Hits in Test** | **28.8** (High danger) | **0.8** (97% safer) | **0.4** (98% safer) | **10.5** (Moderate) |
| **Hazard Recall** | High | Cautious | Swift | Robust |
| **Key Takeaway** | Maximizes path efficiency but steps into hazards | Learns safe buffer due to $\epsilon$-greedy anticipation | Best team dispersion | Scales to continuous observations |

### Why this happens (Core RL Insight for your Course):
- **Q-Learning** uses $\max_{a'} Q(s', a')$ for its update target. It assumes optimal actions will always be taken in the future, meaning it does not penalize paths directly adjacent to hazardous zones. When $\epsilon$-greedy exploratory actions occur, it accidentally steps into hazards frequently.
- **SARSA** evaluates the action actually taken $Q(s', a')$. It realizes that near a hazard, a random exploratory move has a high chance of incurring a severe $-20$ penalty. Thus, SARSA naturally learns a **protective safety margin** around hazards!

---

## 📖 Theoretical Reference
For complete mathematical derivations, Bellman update equations, state-action formulations, and Dec-POMDP definitions, read [SWARM_RL_PROJECT_DOCUMENT.md](file:///Users/afnankaif/Desktop/rein/SWARM_RL_PROJECT_DOCUMENT.md).
# reinforcement-learning-
