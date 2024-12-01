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

from sac_mppi.brax_rl.brax_env import UnitreeGo2EnvRL
from sac_mppi.rsl_rl.rsl_rl.env.vec_env import VecEnv
import torch.utils.dlpack as tpack
from jax._src.dlpack import from_dlpack, to_dlpack
from brax import base
import time

def jax_to_torch(tensor: jax.Array) -> torch.Tensor:
    """Converts a jax tensor to a torch tensor without copying from the GPU."""
    tensor = to_dlpack(tensor)
    return tpack.from_dlpack(tensor)


def torch_to_jax(tensor: torch.Tensor) -> jax.Array:
    """Converts a torch tensor to a jax tensor without copying from the GPU."""
    tensor = tpack.to_dlpack(tensor)
    return from_dlpack(tensor)

class BraxRslGo2Env(VecEnv):
    def __init__(self, num_envs=1, *args, **kargs):

        self.env = UnitreeGo2EnvRL()
        
        self.jit_step = jax.jit(jax.vmap(self.env.step_rsl))
        self.jit_reset = jax.jit(jax.vmap(self.env.reset))
        
        self.num_envs = num_envs
        
        self.key = jax.random.PRNGKey(123)
        self.key, key_use = jax.random.split(self.key, 2)
        self.num_actions = self.env.num_actions
        self.num_obs = self.env.reset(key_use).obs.shape[0]  
        self.max_episode_length = 1000
        
        
        
        print(f"SACMPPIEnv: num_obs={self.num_obs}, num_actions={self.num_actions}")

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Buffers
        self.obs_buf = torch.zeros((num_envs, self.num_obs), dtype=torch.float32, device=self.device, requires_grad=False)
        self.rew_buf = torch.zeros((num_envs, 1), dtype=torch.float32, device=self.device, requires_grad=False)
        self.reset_buf = torch.zeros((num_envs, 1), dtype=torch.float32, device=self.device, requires_grad=False)
        self.episode_length_buf = torch.zeros((num_envs, 1), device=self.device, requires_grad=False)
        self.episode_sums = {
            "tracking_lin_vel": torch.zeros((num_envs, 1), dtype=torch.float32, device=self.device, requires_grad=False),
            "tracking_ang_vel": torch.zeros((num_envs, 1), dtype=torch.float32, device=self.device, requires_grad=False),
            "lin_vel_z": torch.zeros((num_envs, 1), dtype=torch.float32, device=self.device, requires_grad=False),
            "ang_vel_xy": torch.zeros((num_envs, 1), dtype=torch.float32, device=self.device, requires_grad=False),
            "orientation": torch.zeros((num_envs, 1), dtype=torch.float32, device=self.device, requires_grad=False),
            "torques": torch.zeros((num_envs, 1), dtype=torch.float32, device=self.device, requires_grad=False),
            "action_rate": torch.zeros((num_envs, 1), dtype=torch.float32, device=self.device, requires_grad=False),
            "feet_air_time": torch.zeros((num_envs, 1), dtype=torch.float32, device=self.device, requires_grad=False),
            "stand_still": torch.zeros((num_envs, 1), dtype=torch.float32, device=self.device, requires_grad=False),
            "termination": torch.zeros((num_envs, 1), dtype=torch.float32, device=self.device, requires_grad=False),
            "foot_slip": torch.zeros((num_envs, 1), dtype=torch.float32, device=self.device, requires_grad=False),
        }
        self.extras = {}

        # State trackers
        self.states = None

        self.reset()
        
        
    # @property
    # def joint_range(self):
    #     return self.env.joint_range
    
    @property
    def sys(self):
        return self.env.sys
    
    @property
    def dt(self):
        return self.env._dt

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
        
        keys = jax.random.split(self.key, self.num_envs+1)
        self.key, key_list = keys[0], keys[1:]
        
        self.states = self.jit_reset(key_list)
        
        
        self.obs_buf[:] = jax_to_torch(self.states.obs)
        self.episode_length_buf[:] = 0
        self.reset_buf[:] = 0
        self.extras = {
            "observations": {},
            "episode": {},
        }

        return self.obs_buf, self.extras

    
    def reset_idx(self, ids: list[int]) -> tuple[torch.Tensor, dict]:
        if ids.shape[0] == 0:
            return
        
        ids_torch = jax_to_torch(ids)
        self.episode_length_buf[ids_torch] = 0
        # import ipdb; ipdb.set_trace()
        for key in self.episode_sums.keys():
            self.extras["episode"][key] = torch.mean(self.episode_sums[key][ids_torch]) / (torch.max(self.episode_length_buf[ids_torch]) + 1)
            # print("key", key, self.episode_sums[key][ids_torch])
            self.episode_sums[key][ids_torch] = 0.

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
        actions = actions.to(self.device)
        
        actions_jax = torch_to_jax(actions)
        # print("actions", actions)
        self.states = self.jit_step(self.states, actions_jax)
        
        self.obs_buf[:] = jax_to_torch(self.states.obs)
        self.rew_buf[:, 0] = jax_to_torch(self.states.reward)
        self.reset_buf[:, 0] = jax_to_torch(self.states.done)
        self.episode_length_buf[:] += 1
       
        reset_idx = torch.nonzero(self.reset_buf)[:, 0]
        reset_idx = torch_to_jax(reset_idx)

        self.reset_idx(reset_idx)

        info = self.states.info
        
        
        for key in self.episode_sums.keys():
            self.episode_sums[key][:, 0] += jax_to_torch(info["rewards"][key])
       
        return (
            self.obs_buf,
            self.rew_buf.squeeze(1),
            self.reset_buf.squeeze(1),
            self.extras,
        )