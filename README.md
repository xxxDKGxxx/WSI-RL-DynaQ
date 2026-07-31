# DynaQ & Reinforcement Learning in Slippery GridWorld

<p align="left">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+"></a>
  <a href="https://jupyter.org/"><img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter Notebook"></a>
  <a href="https://numpy.org/"><img src="https://img.shields.io/badge/NumPy-2.2-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge" alt="MIT License"></a>
  <img src="https://img.shields.io/badge/Course-WSI--RL-purple.svg?style=for-the-badge" alt="Course WSI-RL">
</p>

DynaQ-GridWorld is a Reinforcement Learning (RL) benchmark and visualization platform focused on evaluating model-based and model-free RL algorithms in custom GridWorld environments with stochastic transition dynamics and dynamic target movements. The project highlights the **Dyna-Q** architecture, demonstrating how planning with a learned environment model significantly accelerates convergence compared to standard model-free temporal difference methods.

This project was developed for the **"Wstęp do Sztucznej Inteligencji" (Introduction to Artificial Intelligence)** course at the Faculty of Mathematics and Information Science (MiNI), Warsaw University of Technology.

## Core Features

- **Stochastic Dyna-Q Implementation**: Model-based RL agent integrating model-free Q-learning with simulated experience replay using a learned probabilistic transition model (`StochasticDynaQModel`).
- **Diverse GridWorld Environments**:
  - **Slippery GridWorld**: Custom tabular grid with action execution stochasticity (slipping probabilities).
  - **Cliff GridWorld**: Environment with high-penalty cliff states requiring careful risk-averse navigation.
  - **Dynamic Target Environments**: GridWorld featuring moving targets with both stochastic and deterministic movement patterns.
  - **Multiple Targets**: Environment supporting multi-goal destinations.
- **Algorithm Benchmarking & Comparison**: Empirical comparison between model-based approaches (**Dyna-Q**, **Value Iteration**) and model-free algorithms (**Q-Learning**, **SARSA**).
- **Rich Visualizations & Media Generation**:
  - Value function heatmaps (`value_*.png`).
  - Learned policy arrow plots (`policy_*.png`).
  - Animated trajectory GIFs of agent episodes (`episode_*.gif`).
- **Interactive Jupyter Notebooks**: Comprehensive suite of notebooks for conducting experiments, tuning hyperparameters, and visualizing policy convergence.

## System Architecture & Environments

### Implemented Algorithms

1. **Dyna-Q**: Combines direct RL (Q-Learning) with model-based planning by sampling state-action transitions from an online learned environment model.
2. **Value Iteration (VI)**: Dynamic programming baseline for exact optimal policy computation given full environment transition probabilities.
3. **Q-Learning**: Off-policy temporal difference (TD) control algorithm.
4. **SARSA**: On-policy temporal difference (TD) control algorithm.

### Key Environments

- `SlipperyGridWorld` (`helpers/env.py`): Base gridworld with slip probabilities.
- `CliffGridWorld` (`helpers/cliff_env.py`): Gridworld containing designated cliff tiles.
- `DynamicTargetGridWorld` (`helpers/dynamic_target_env.py`): Environment where target coordinates update dynamically.
- `MultipleTargetsGridWorld` (`helpers/multiple_targets_env.py`): Environment featuring multiple goal coordinates.

## Tech Stack

- **Language**: Python 3.10+
- **Numerical Computing**: NumPy
- **Data Visualization & Graphics**: Matplotlib, ImageIO
- **Interactive Environment**: Jupyter Notebook
- **Output Artifacts**: Animated GIFs, static plot graphics, PDF presentation

## Getting Started

### Prerequisites

- [Python 3.10+](https://www.python.org/)
- [Jupyter Notebook / Lab](https://jupyter.org/)

### Installation & Quick Start

1. **Clone the repository**:
   ```bash
   git clone https://github.com/xxxDKGxxx/WSI-RL-LastProject.git
   cd WSI-RL-LastProject
   ```

2. **Create and activate a virtual environment** (optional but recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run Experiments**:
   Launch Jupyter Notebook to interactively run the experiments:
   ```bash
   jupyter notebook
   ```
   Explore the notebooks in the `Notebooks/` directory:
   - `SlipperyGridWorldClean.ipynb`: Standard Dyna-Q and Q-learning benchmarking.
   - `SlipperyGridWorldCliff.ipynb`: Cliff walking environment analysis.
   - `SlipperyGridWorldDynamicTarget.ipynb`: Dynamic target tracking experiments.

## Project Structure

```
DynaQProject/
├── dynaq/                  # Dyna-Q algorithm implementation & model definition
│   └── dyna_q.py
├── helpers/                # GridWorld environment implementations & visualizers
│   ├── base_env.py
│   ├── cliff_env.py
│   ├── dynamic_target_env.py
│   ├── dynamic_deterministic_target_env.py
│   ├── env.py
│   ├── multiple_targets_env.py
│   └── viz.py
├── Notebooks/              # Jupyter notebooks for experiments & evaluation
├── Results/                # Saved benchmark evaluation metrics & output artifacts
├── visuals/                # Rendered visual outputs & frame assets
├── Prezentacja_WSI_dynaq.pdf # Project presentation PDF
└── requirements.txt        # Python package dependencies
```

## Authors

- [Dominik Zieliński](https://github.com/xxxDKGxxx)
- [Artur Szabelski](https://github.com/Artur112233)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
