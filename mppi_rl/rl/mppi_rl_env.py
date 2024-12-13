#!/usr/bin/env python3
#
# Created on Tue Nov 26 2024 16:58:43
# Author: Mukai (Tom Notch) Yu
# Email: mukaiy@andrew.cmu.edu
# Affiliation: Carnegie Mellon University, Robotics Institute
#
# Copyright Ⓒ 2024 Mukai (Tom Notch) Yu
#
import jax
import jax.numpy as jnp
import torch

from mppi_rl.dial_mpc.dial_mpc.envs.unitree_go2_env import UnitreeGo2Env
from mppi_rl.dial_mpc.dial_mpc.envs.unitree_go2_env import UnitreeGo2EnvConfig
from mppi_rl.rsl_rl.rsl_rl.env.vec_env import VecEnv


class SACMPPIEnv(VecEnv):
    def __init__(self, config: dict, num_envs=1, *args, **kargs):
        # Construct UnitreeGo2EnvConfig
        # <TODO> Some remapping is needed here as UnitreeGo2EnvConfig and BaseEnvConfig
        # expect only a subset of keys in the config yaml file.
        self.env_config = UnitreeGo2EnvConfig(**config)

        self.envs = [
            UnitreeGo2Env(config) for _ in range(num_envs)
        ]  # Multiple environments
        self.num_envs = num_envs
        self.num_obs = (
            self.envs[0]
            ._get_obs(self.envs[0].reset(jax.random.PRNGKey(0)), {})
            .shape[0]
        )
        self.num_actions = self.envs[0].action_space.shape[0]
        self.max_episode_length = 1000

        # Buffers
        self.obs_buf = torch.zeros((num_envs, self.num_obs))
        self.rew_buf = torch.zeros((num_envs, 1))
        self.reset_buf = torch.zeros((num_envs, 1))
        self.episode_length_buf = torch.zeros((num_envs, 1))
        self.extras = {}

        # State trackers
        self.states = [
            env.reset(jax.random.PRNGKey(i)) for i, env in enumerate(self.envs)
        ]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        """Return the current observations.

        Returns:
            Tuple[torch.Tensor, dict]: Tuple containing the observations and extras.
        """
        return self.obs_buf.to(self.device), self.extras

    def reset(self) -> tuple[torch.Tensor, dict]:
        """Reset all environment instances.

        Returns:
            Tuple[torch.Tensor, dict]: Tuple containing the observations and extras.
        """
        for i in range(self.num_envs):
            self.states[i] = self.envs[i].reset(jax.random.PRNGKey(i))
            obs = self.states[i].obs
            self.obs_buf[i] = torch.tensor(jnp.array(obs), dtype=torch.float32)
            self.episode_length_buf[i] = 0
            self.reset_buf[i] = 0

        return self.obs_buf.to(self.device), self.extras

    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Apply input action on the environment.

        Args:
            actions (torch.Tensor): Input actions to apply. Shape: (num_envs, num_actions)

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
                A tuple containing the observations, rewards, dones and extra information (metrics).
        """
        for i in range(self.num_envs):
            if self.reset_buf[i]:  # Reset environment if it's done
                self.states[i] = self.envs[i].reset(jax.random.PRNGKey(i))
                self.reset_buf[i] = 0

            # Convert action to JAX array and step the environment
            jax_action = jnp.array(actions[i].cpu().numpy())
            self.states[i] = self.envs[i].step(self.states[i], jax_action)

            # Update buffers
            self.obs_buf[i] = torch.tensor(
                jnp.array(self.states[i].obs), dtype=torch.float32
            )
            self.rew_buf[i] = torch.tensor(
                self.states[i].reward, dtype=torch.float32
            ).unsqueeze(0)
            self.reset_buf[i] = torch.tensor(
                self.states[i].done, dtype=torch.float32
            ).unsqueeze(0)
            self.episode_length_buf[i] += 1

        return (
            self.obs_buf.to(self.device),
            self.rew_buf.to(self.device),
            self.reset_buf.to(self.device),
            self.extras,
        )
