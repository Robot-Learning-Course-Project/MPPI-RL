# Soft Actor Critique Model Predictive Path Integral

[![pre-commit](https://github.com/Robot-Learning-Course-Project/SAC-MPPI/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/Robot-Learning-Course-Project/SAC-MPPI/actions/workflows/pre-commit.yml)

## Installation

1. Clone the repository

   ```Shell
   git clone --recursive git@github.com:Robot-Learning-Course-Project/SAC-MPPI.git
   cd SAC-MPPI
   ```

1. Create an environment, either using conda or virtualenv

   1. Using conda

      ```Shell
      conda env create -f environment.yml
      conda activate sac-mppi
      ```

   1. Using virtualenv

      ```Shell
      python3.10 -m venv .venv
      source .venv/bin/activate
      ```

   Install packages

   ```Shell
   pip install -e .

   cd sac_mppi/dial_mpc
   pip install -e .

   cd sac_mppi/rsl_rl
   pip install -e .
   ```

1. Run example:

   ```Shell
   sac-mppi -c ./config/default.yaml
   ```

## Run our method

1. train an RL policy

```
cd sac_mppi/
python scripts/brax/train.py
```

1. visualize the RL policy

copy model path `logs/.../policy_stepxxx` to `play.py`
```
python scripts/brax/play.py
```

1. Run our method

copy model relative path `brax_go2/.../value_stepxxx` to `sac_mppi/dial_mpc/dial_mpc/examples/unitree_go2_trot_hybrid.yaml`
```
python sac_mppi/dial_mpc/dial_mpc/core/dial_custom_hybrid.py --example unitree_go2_trot_hybrid
```


## Developer Quick Start

- Run [scripts/dev-setup.sh](scripts/dev-setup.sh) to setup the development environment

