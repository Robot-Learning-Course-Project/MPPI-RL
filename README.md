# Model Predictive Path Integral with Reinforcement Learning baseline

[![pre-commit](https://github.com/Robot-Learning-Course-Project/MPPI-RL/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/Robot-Learning-Course-Project/MPPI-RL/actions/workflows/pre-commit.yml)

## Installation

1. Clone the repository

   ```Shell
   git clone --recursive git@github.com:Robot-Learning-Course-Project/MPPI-RL.git
   cd MPPI-RL
   ```

1. Create an environment, either using conda or virtualenv

   1. Using conda

      ```Shell
      conda env create -f environment.yml
      conda activate mppi-rl
      ```

   1. Using virtualenv

      ```Shell
      python3.10 -m venv .venv
      source .venv/bin/activate
      ```

   Install packages

   ```Shell
   pip install -e .

   cd mppi_rl/dial_mpc
   pip install -e .
   ```

## Run our method

1. train an RL policy

   ```Shell
   mppi_rl/scripts/brax/train.py
   ```

1. visualize the RL policy

   copy model path `logs/.../policy_stepxxx` to `play.py`

   ```Shell
   mppi_rl/scripts/brax/play.py
   ```

1. Run our method

   copy model relative path `brax_go2/.../value_stepxxx` to `mppi_rl/dial_mpc/dial_mpc/examples/unitree_go2_trot_hybrid.yaml`

   ```Shell
   python mppi_rl/dial_mpc/dial_mpc/core/dial_custom_hybrid.py --example unitree_go2_trot_hybrid
   ```

## Developer Quick Start

- Run [scripts/dev-setup.sh](scripts/dev-setup.sh) to setup the development environment
