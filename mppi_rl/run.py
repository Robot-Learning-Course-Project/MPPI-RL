#!/usr/bin/env python3
#
# Created on Wed Nov 27 2024 00:25:51
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute
#
# Copyright Ⓒ 2024 Mukai (Tom Notch) Yu
#
import argparse

from mppi_rl.rl.mppi_rl_env import SACMPPIEnv
from mppi_rl.rsl_rl.rsl_rl.runners.on_policy_runner import OnPolicyRunner
from mppi_rl.utils.files import print_dict
from mppi_rl.utils.files import read_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", "-c", type=str, required=True, help="Path to the config file."
    )
    args = parser.parse_args()

    config = read_file(args.config)
    print_dict(config)

    env = SACMPPIEnv(config["env_config"])
    runner = OnPolicyRunner(env, config["runner_config"])

    runner.learn(num_learning_iterations=config["runner_config"]["runner"]["num_iter"])


if __name__ == "__main__":
    main()
