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

from sac_mppi.dial_mpc.dial_mpc.envs.unitree_go2_env import UnitreeGo2Env
from sac_mppi.dial_mpc.dial_mpc.envs.unitree_go2_env import UnitreeGo2EnvConfig
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

class SACMPPIEnv(VecEnv):
    def __init__(self, config: dict, num_envs=1, *args, **kargs):
        # Construct UnitreeGo2EnvConfig
        # <TODO> Some remapping is needed here as UnitreeGo2EnvConfig and BaseEnvConfig
        # expect only a subset of keys in the config yaml file.

        self.env_config = UnitreeGo2EnvConfig(config)

        self.env = UnitreeGo2Env(config)
        
        print("joint_range", self.env.joint_range)
        # import ipdb; ipdb.set_trace()
        # Need 
        # 1. step_jit_vmap
        # 2. reset_jit_vmap
        self.jit_step = jax.jit(jax.vmap(self.env.step))
        # self.jit_step = jax.jit(self.env.step)
        self.jit_reset = jax.jit(jax.vmap(self.env.reset))
        # self.jit_reset = jax.jit(self.env.reset)
        # self.jit_reset_idx = jax.jit(self.reset_idx)
        
        self.num_envs = num_envs
        
        self.key = jax.random.PRNGKey(123)
        self.key, key_use = jax.random.split(self.key, 2)
        self.num_actions = self.env.num_actions
        self.num_obs = self.env.reset(key_use).obs.shape[0] + self.num_actions  
        self.max_episode_length = 1000
        
        
        
        print(f"SACMPPIEnv: num_obs={self.num_obs}, num_actions={self.num_actions}")

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Buffers
        self.obs_buf = torch.zeros((num_envs, self.num_obs), dtype=torch.float32, device=self.device, requires_grad=False)
        self.rew_buf = torch.zeros((num_envs, 1), dtype=torch.float32, device=self.device, requires_grad=False)
        self.reset_buf = torch.zeros((num_envs, 1), dtype=torch.float32, device=self.device, requires_grad=False)
        self.episode_length_buf = torch.zeros((num_envs, 1), device=self.device, requires_grad=False)
        self.episode_sums = {
            "reward_gaits": torch.zeros((num_envs, 1), dtype=torch.float32, device=self.device, requires_grad=False),
            "reward_air_time": torch.zeros((num_envs, 1), dtype=torch.float32, device=self.device, requires_grad=False),
            "reward_pos": torch.zeros((num_envs, 1), dtype=torch.float32, device=self.device, requires_grad=False),
            "reward_upright": torch.zeros((num_envs, 1), dtype=torch.float32, device=self.device, requires_grad=False),
            "reward_yaw": torch.zeros((num_envs, 1), dtype=torch.float32, device=self.device, requires_grad=False),
            "reward_vel": torch.zeros((num_envs, 1), dtype=torch.float32, device=self.device, requires_grad=False),
            "reward_ang_vel": torch.zeros((num_envs, 1), dtype=torch.float32, device=self.device, requires_grad=False),
            "reward_height": torch.zeros((num_envs, 1), dtype=torch.float32, device=self.device, requires_grad=False),
            "reward_energy": torch.zeros((num_envs, 1), dtype=torch.float32, device=self.device, requires_grad=False),
            "reward_alive": torch.zeros((num_envs, 1), dtype=torch.float32, device=self.device, requires_grad=False),
            "reward_dof_vel": torch.zeros((num_envs, 1), dtype=torch.float32, device=self.device, requires_grad=False),
            "reward_dof_pos_limit": torch.zeros((num_envs, 1), dtype=torch.float32, device=self.device, requires_grad=False),
            "reward_ctrl_rate": torch.zeros((num_envs, 1), dtype=torch.float32, device=self.device, requires_grad=False),
        }
        self.last_actions = torch.zeros((num_envs, self.num_actions), dtype=torch.float32, device=self.device, requires_grad=False)
        self.extras = {}

        # State trackers
        self.states = None

        self.reset()
        
        
    @property
    def joint_range(self):
        return self.env.joint_range
    
    @property
    def sys(self):
        return self.env.sys
    
    @property
    def dt(self):
        return self.env.dt

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
        
        self.last_actions[:] = 0
        
        obs_env = jax_to_torch(self.states.obs)
        obs = torch.cat([obs_env, self.last_actions], dim=1)
        self.obs_buf[:] = obs
        self.episode_length_buf[:] = 0
        self.reset_buf[:] = 0
        self.extras = {
            "observations": {},
            "episode": {},
        }

        return self.obs_buf, self.extras

    
    def reset_idx_bak(self, states, ids: list[int]) -> tuple[torch.Tensor, dict]:
        
        if ids.shape[0] == 0:
            return states
        
        # print("device", ids.device)
        
        # print("reset_idx", ids)
        keys = jax.random.split(self.key, len(ids)+1)
        self.key, key_list = keys[0], keys[1:]
        reseted_states = self.jit_reset(key_list)
        
        ids_torch = jax_to_torch(ids)
        self.episode_length_buf[ids_torch] = 0
        self.last_actions[ids_torch] = 0
        # self.reset_buf[ids_torch] = 0
        
        obs = states.obs
        reward = states.reward
        done = states.done
        info = states.info
        pipeline_state = states.pipeline_state
        
        # pipeline_state cared about
        # ctrl, qpos, qvel, x, xd, q, site_xpos, contact
        # print(f"obs device: {obs.device}, ids device: {ids.device}, resstates.obs device: {reseted_states.obs.device}")
        obs = obs.at[ids].set(reseted_states.obs)
        reward = reward.at[ids].set(reseted_states.reward)
        done = done.at[ids].set(reseted_states.done)
        
        for info_key in info.keys():
            info[info_key] = info[info_key].at[ids].set(reseted_states.info[info_key])
        
       
        ps_ctrl = pipeline_state.ctrl
        ps_qpos = pipeline_state.qpos
        ps_qvel = pipeline_state.qvel
        ps_q = pipeline_state.q
        ps_site_xpos = pipeline_state.site_xpos
        ps_contact = pipeline_state.contact
        ps_x = pipeline_state.x
        ps_xd = pipeline_state.xd
        
        
        # import ipdb; ipdb.set_trace()
        ps_x = ps_x.replace(
            pos=ps_x.pos.at[ids].set(reseted_states.pipeline_state.x.pos),
            rot=ps_x.rot.at[ids].set(reseted_states.pipeline_state.x.rot),
        )
        ps_xd = ps_xd.replace(
            vel=ps_xd.vel.at[ids].set(reseted_states.pipeline_state.xd.vel),
            ang=ps_xd.ang.at[ids].set(reseted_states.pipeline_state.xd.ang),
        )
        ps_ctrl = ps_ctrl.at[ids].set(reseted_states.pipeline_state.ctrl)
        ps_qpos = ps_qpos.at[ids].set(reseted_states.pipeline_state.qpos)
        ps_qvel = ps_qvel.at[ids].set(reseted_states.pipeline_state.qvel)
        ps_q = ps_q.at[ids].set(reseted_states.pipeline_state.q)
        ps_site_xpos = ps_site_xpos.at[ids].set(reseted_states.pipeline_state.site_xpos)
        
        # efc_address = ps_contact.efc_address
        # import ipdb; ipdb.set_trace()
        # efc_address[ids] = reseted_states.pipeline_state.contact.efc_address
        # link_idx = ps_contact.link_idx
        # for i in ids: link_idx[i] = reseted_states.pipeline_state.contact.link_idx[i]
        ps_contact = ps_contact.replace(
            dist = ps_contact.dist.at[ids].set(reseted_states.pipeline_state.contact.dist),
            pos = ps_contact.pos.at[ids].set(reseted_states.pipeline_state.contact.pos),
            frame = ps_contact.frame.at[ids].set(reseted_states.pipeline_state.contact.frame),
            includemargin = ps_contact.includemargin.at[ids].set(reseted_states.pipeline_state.contact.includemargin),
            friction=ps_contact.friction.at[ids].set(reseted_states.pipeline_state.contact.friction),
            solref=ps_contact.solref.at[ids].set(reseted_states.pipeline_state.contact.solref),
            solreffriction=ps_contact.solreffriction.at[ids].set(reseted_states.pipeline_state.contact.solreffriction),
            solimp=ps_contact.solimp.at[ids].set(reseted_states.pipeline_state.contact.solimp),
            geom1=ps_contact.geom1.at[ids].set(reseted_states.pipeline_state.contact.geom1),
            geom2=ps_contact.geom2.at[ids].set(reseted_states.pipeline_state.contact.geom2),
            geom=ps_contact.geom.at[ids].set(reseted_states.pipeline_state.contact.geom),
            # efc_address=ps_contact.efc_address.at[ids].set(reseted_states.pipeline_state.contact.efc_address),
            # link_idx=ps_contact.link_idx.at[ids].set(reseted_states.pipeline_state.contact.link_idx),
            elasticity=ps_contact.elasticity.at[ids].set(reseted_states.pipeline_state.contact.elasticity),
        )
        
        pipeline_state = pipeline_state.replace(
            ctrl=ps_ctrl,
            qpos=ps_qpos,
            qvel=ps_qvel,
            x=ps_x,
            xd=ps_xd,
            q=ps_q,
            site_xpos=ps_site_xpos,
            contact=ps_contact,
        )
        
        states = states.replace(
            obs=obs,
            reward=reward,
            done=done,
            info=info,
            pipeline_state=pipeline_state,
        )
        
        for key in self.episode_sums.keys():
            self.extras["episode"][key] = torch.mean(self.episode_sums[key][ids_torch])
            self.episode_sums[key][ids_torch] = 0.
            
        return states
    
    def reset_idx(self, ids: list[int]) -> tuple[torch.Tensor, dict]:
        if ids.shape[0] == 0:
            return
        
        ids_torch = jax_to_torch(ids)
        self.episode_length_buf[ids_torch] = 0
        self.last_actions[ids_torch] = 0
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
        
        actions = torch.clip(actions, -1.0, 1.0)
        actions_jax = torch_to_jax(actions)
        # print("actions", actions)
        self.states = self.jit_step(self.states, actions_jax)
        
        obs_env = jax_to_torch(self.states.obs)
        self.rew_buf[:, 0] = jax_to_torch(self.states.reward)
        self.reset_buf[:, 0] = jax_to_torch(self.states.done)
        self.episode_length_buf[:] += 1
        obs = torch.concatenate([obs_env, self.last_actions], dim=1)
        self.obs_buf[:] = obs
        self.last_actions[:] = actions.clone()
       
        reset_idx = torch.nonzero(self.reset_buf)[:, 0]
        reset_idx = torch_to_jax(reset_idx)

        self.reset_idx(reset_idx)

        info = self.states.info
        
        for key in self.episode_sums.keys():
            if "reward" in key:
                self.episode_sums[key][:, 0] += jax_to_torch(info[key])
       
        # self.obs_buf[torch.where(torch.isnan(self.obs_buf))] = 0.0
        # self.rew_buf[torch.where(torch.isnan(self.rew_buf))] = -100.
        return (
            self.obs_buf,
            self.rew_buf.squeeze(1),
            self.reset_buf.squeeze(1),
            self.extras,
        )
