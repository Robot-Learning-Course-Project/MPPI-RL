#!/usr/bin/env python3
import functools
import os
from functools import partial
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from brax import base
from brax import math
from brax.base import Base
from brax.base import Motion
from brax.base import State as PipelineState
from brax.base import Transform
from brax.envs.base import Env
from brax.envs.base import PipelineEnv
from brax.envs.base import State
from brax.io import html
from brax.io import mjcf
from brax.io import model
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from dial_mpc.utils.function_utils import get_foot_step
from dial_mpc.utils.function_utils import global_to_body_velocity
from dial_mpc.utils.io_utils import get_model_path
from jax import numpy as jp
from ml_collections import config_dict

import sac_mppi.dial_mpc.dial_mpc.envs.unitree_go2_env as dial_envs
from sac_mppi.brax_rl import brax_utils
from sac_mppi.dial_mpc.dial_mpc.utils.function_utils import global_to_body_velocity

# @title Import MuJoCo, MJX, and Brax
# GO2_ROOT_PATH = epath.Path('mujoco_menagerie/unitree_go2')


# More legible printing from numpy.
np.set_printoptions(precision=3, suppress=True, linewidth=100)


def get_config():
    """Returns reward config for parkour quadruped environment."""

    def get_default_rewards_config():
        default_config = config_dict.ConfigDict(
            dict(
                # The coefficients for all reward terms used for training. All
                # physical quantities are in SI units, if no otherwise specified,
                # i.e. joint positions are in rad, positions are measured in meters,
                # torques in Nm, and time in seconds, and forces in Newtons.
                scales=config_dict.ConfigDict(
                    dict(
                        # Tracking rewards are computed using exp(-delta^2/sigma)
                        # sigma can be a hyperparameters to tune.
                        # Track the base x-y velocity (no z-velocity tracking.)
                        tracking_lin_vel=1.5 * 1.0,
                        # Track the angular velocity along z-axis, i.e. yaw rate.
                        tracking_ang_vel=0.8,
                        # Below are regularization terms, we roughly divide the
                        # terms to base state regularizations, joint
                        # regularizations, and other behavior regularizations.
                        # Penalize the base velocity in z direction, L2 penalty.
                        lin_vel_z=-2.0,
                        # Penalize the base roll and pitch rate. L2 penalty.
                        ang_vel_xy=-0.05,
                        # Penalize non-zero roll and pitch angles. L2 penalty.
                        orientation=-5.0,
                        # L2 regularization of joint torques, |tau|^2.
                        torques=-0.0002,
                        # Penalize the change in the action and encourage smooth
                        # actions. L2 regularization |action - last_action|^2
                        action_rate=-0.01,
                        # Encourage long swing steps.  However, it does not
                        # encourage high clearances.
                        feet_air_time=0.2,
                        # Encourage no motion at zero command, L2 regularization
                        # |q - q_default|^2.
                        stand_still=-0.5,
                        # Early termination penalty.
                        termination=-1.0,
                        # Penalizing foot slipping on the ground.
                        foot_slip=-0.1,
                        # gaits, only used for deploy
                        gaits=0.0,
                        energy=0.0,
                        height=0.0,
                        upright=0.0,
                        vel=0.0,
                        ang_vel=0.0,
                        yaw=0.0,
                    )
                ),
                # Tracking reward = exp(-error^2/sigma).
                tracking_sigma=0.25,
            )
        )
        return default_config

    default_config = config_dict.ConfigDict(
        dict(
            rewards=get_default_rewards_config(),
        )
    )

    return default_config


class UnitreeGo2EnvRL(PipelineEnv):
    """Environment for training the barkour quadruped joystick policy in MJX."""

    def __init__(
        self,
        deploy: bool = False,
        dial_action_space: bool = False,
        obs_noise: float = 0.05,
        action_scale: float = 0.3,
        get_value: bool = False,
        kick_vel: float = 0.05,
        scene_file: str = "scene_mjx.xml",
        **kwargs,
    ):
        # path = GO2_ROOT_PATH / scene_file
        # sys = mjcf.load(path.as_posix())
        # model_path = get_model_path("unitree_go2", "mjx_scene.xml")
        model_path = get_model_path("unitree_go2", "mjx_scene_position.xml")
        # model_path = get_model_path("unitree_go2", "mjx_scene_force.xml")
        sys = mjcf.load(model_path)
        self.deploy = deploy
        self.get_value = get_value
        self.dial_action_space = dial_action_space
        self._dt = 0.02  # this environment is 50 fps
        sys = sys.tree_replace({"opt.timestep": 0.02})

        # override menagerie params for smoother policy
        sys = sys.replace(
            dof_damping=sys.dof_damping.at[6:].set(0.5239),
            actuator_gainprm=sys.actuator_gainprm.at[:, 0].set(35.0),
            actuator_biasprm=sys.actuator_biasprm.at[:, 1].set(-35.0),
        )

        n_frames = kwargs.pop("n_frames", int(self._dt / sys.opt.timestep))
        super().__init__(sys, backend="mjx", n_frames=n_frames)

        self.reward_config = get_config()

        if self.get_value:
            make_networks_factory = functools.partial(
                ppo_networks.make_ppo_networks,
                policy_hidden_layer_sizes=(512, 256, 128),
            )

            ppo_network = make_networks_factory(
                960, 12, preprocess_observations_fn=running_statistics.normalize
            )
            value_model_path = "/home/wenli/SAC-MPPI/sac_mppi/logs/brax_go2/Dec02_14-52-19_walk/go2_value"
            value_params = model.load_params(value_model_path)
            make_value_inference_fn = brax_utils.make_value_inference_fn(ppo_network)
            value_inference_fn = make_value_inference_fn(value_params)
            self.jit_value_inference_fn = jax.jit(value_inference_fn)

        if self.deploy:
            # self.reward_config.rewards.scales.tracking_lin_vel = 1.
            # self.reward_config.rewards.scales.tracking_ang_vel = .2
            # self.reward_config.rewards.scales.lin_vel_z = -1.
            # self.reward_config.rewards.scales.ang_vel_xy = -0.01
            # self.reward_config.rewards.scales.action_rate = -0.01
            # self.reward_config.rewards.scales.torques = -0.00
            # self.reward_config.rewards.scales.termination = -0.
            # self.reward_config.rewards.scales.orientation = -0.05
            # self.reward_config.rewards.scales.stand_still = 0.
            # self.reward_config.rewards.scales.feet_air_time = 0.2
            # self.reward_config.rewards.scales.foot_slip = -0.001
            # self.reward_config.rewards.scales.gaits = 0.1
            # self.reward_config.rewards.scales.energy = 0.1
            # self.reward_config.rewards.scales.height = 1. * 1.
            # self.reward_config.rewards.scales.upright = 1.

            self.reward_config.rewards.scales.tracking_lin_vel = 0.0
            self.reward_config.rewards.scales.tracking_ang_vel = 0.0
            self.reward_config.rewards.scales.lin_vel_z = 0.0
            self.reward_config.rewards.scales.ang_vel_xy = -0.0
            self.reward_config.rewards.scales.action_rate = 0.0
            self.reward_config.rewards.scales.torques = -0.00
            self.reward_config.rewards.scales.termination = -0.0
            self.reward_config.rewards.scales.orientation = -0.0
            self.reward_config.rewards.scales.stand_still = 0.0
            self.reward_config.rewards.scales.feet_air_time = 0.0
            self.reward_config.rewards.scales.foot_slip = -0.005

            self.reward_config.rewards.scales.gaits = 0.1
            self.reward_config.rewards.scales.energy = 0.0
            self.reward_config.rewards.scales.height = 0.5
            self.reward_config.rewards.scales.upright = 0.5
            self.reward_config.rewards.scales.vel = 1.0
            self.reward_config.rewards.scales.ang_vel = 1.0
            self.reward_config.rewards.scales.yaw = 0.3

        # set custom from kwargs
        for k, v in kwargs.items():
            if k.endswith("_scale"):
                self.reward_config.rewards.scales[k[:-6]] = v

        self._torso_idx = mujoco.mj_name2id(
            sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, "torso"
        )
        self._action_scale = action_scale
        self._obs_noise = obs_noise
        self._kick_vel = kick_vel
        self._init_q = jp.array(sys.mj_model.keyframe("home").qpos)
        self._default_pose = sys.mj_model.keyframe("home").qpos[7:]
        # self.lowers = jp.array([-0.8, -1.2, -2.5] * 4)
        # self.uppers = jp.array([0.8, 2.3, -0.6] * 4)
        # [
        #             [-0.5, 0.5],
        #             [0.4, 1.4],
        #             [-2.3, -0.85],
        #             [-0.5, 0.5],
        #             [0.4, 1.4],
        #             [-2.3, -0.85],
        #             [-0.5, 0.5],
        #             [0.4, 1.4],
        #             [-2.3, -1.3],
        #             [-0.5, 0.5],
        #             [0.4, 1.4],
        #             [-2.3, -1.3],
        #         ]
        self.lowers = jp.array([-0.5, 0.4, -2.3] * 4)
        self.uppers = jp.array([0.5, 1.4, -1.3] * 4)
        self.num_actions = 12
        feet_site = [
            "FL_foot",
            "FR_foot",
            "RL_foot",
            "RR_foot",
        ]
        feet_site_id = [
            mujoco.mj_name2id(sys.mj_model, mujoco.mjtObj.mjOBJ_SITE.value, f)
            for f in feet_site
        ]
        assert not any(id_ == -1 for id_ in feet_site_id), "Site not found."
        self._feet_site_id = np.array(feet_site_id)
        lower_leg_body = [
            "FL_calf",
            "FR_calf",
            "RL_calf",
            "RR_calf",
        ]
        lower_leg_body_id = [
            mujoco.mj_name2id(sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, l)
            for l in lower_leg_body
        ]
        assert not any(id_ == -1 for id_ in lower_leg_body_id), "Body not found."
        self._lower_leg_body_id = np.array(lower_leg_body_id)
        self._foot_radius = 0.0175
        self._nv = sys.nv

        self._gait = "canter"
        # self._gait = "walk"
        self._gait_phase = {
            "stand": jnp.zeros(4),
            "walk": jnp.array([0.0, 0.5, 0.75, 0.25]),
            "trot": jnp.array([0.0, 0.5, 0.5, 0.0]),
            "canter": jnp.array([0.0, 0.33, 0.33, 0.66]),
            "gallop": jnp.array([0.0, 0.05, 0.4, 0.35]),
        }
        self._gait_params = {
            #                  ratio, cadence, amplitude
            "stand": jnp.array([1.0, 1.0, 0.0]),
            "walk": jnp.array([0.75, 1.0, 0.08]),
            "trot": jnp.array([0.45, 2, 0.08]),
            "canter": jnp.array([0.4, 4, 0.06]),
            "gallop": jnp.array([0.3, 3.5, 0.10]),
        }

    @partial(jax.jit, static_argnums=(0,))
    def act2joint(self, act: jax.Array) -> jax.Array:
        act_normalized = (act * self._action_scale + 1.0) / 2.0  # normalize to [0, 1]
        joint_targets = self.lowers + act_normalized * (
            self.uppers - self.lowers
        )  # scale to joint range
        joint_targets = jnp.clip(
            joint_targets,
            self.lowers,
            self.uppers,
        )
        return joint_targets

    def sample_command(self, rng: jax.Array) -> jax.Array:
        if self.deploy:
            lin_vel_x = [0.8, 0.8]
            lin_vel_y = [-0.0, 0.0]
            ang_vel_yaw = [-0.0, 0.0]
        else:
            # lin_vel_x = [0.8, 0.8]
            # lin_vel_y = [-0., 0.]
            # ang_vel_yaw = [-0., 0.]
            lin_vel_x = [-0.5, 1.0]  # min max [m/s]
            lin_vel_y = [-0.5, 0.5]  # min max [m/s]
            ang_vel_yaw = [-0.5, 0.5]  # min max [rad/s]

        # lin_vel_y = [-0.8, 0.8]  # min max [m/s]
        # lin_vel_y = [-0., 0.]  # min max [m/s]
        # ang_vel_yaw = [-0.7, 0.7]  # min max [rad/s]
        # ang_vel_yaw = [-0., 0.]  # min max [rad/s]

        _, key1, key2, key3 = jax.random.split(rng, 4)
        lin_vel_x = jax.random.uniform(
            key1, (1,), minval=lin_vel_x[0], maxval=lin_vel_x[1]
        )
        lin_vel_y = jax.random.uniform(
            key2, (1,), minval=lin_vel_y[0], maxval=lin_vel_y[1]
        )
        ang_vel_yaw = jax.random.uniform(
            key3, (1,), minval=ang_vel_yaw[0], maxval=ang_vel_yaw[1]
        )
        new_cmd = jp.array([lin_vel_x[0], lin_vel_y[0], ang_vel_yaw[0]])
        return new_cmd

    def reset(self, rng: jax.Array) -> State:  # pytype: disable=signature-mismatch
        rng, key = jax.random.split(rng)

        pipeline_state = self.pipeline_init(self._init_q, jp.zeros(self._nv))

        state_info = {
            "rng": rng,
            "last_act": jp.zeros(12),
            "last_ctrl": jp.zeros(12),
            "last_vel": jp.zeros(12),
            "command": self.sample_command(key),
            "last_contact": jp.zeros(4, dtype=bool),
            "feet_air_time": jp.zeros(4),
            "rewards": {k: 0.0 for k in self.reward_config.rewards.scales.keys()},
            "kick": jp.array([0.0, 0.0]),
            "step": 0,
            "vel_tar": jnp.array([0.0, 0.0, 0.0]),
            "ang_vel_tar": jnp.array([0.0, 0.0, 0.0]),
        }
        if self.get_value:
            state_info.update(
                {
                    "value": 0.0,
                }
            )
        obs_history = jp.zeros(15 * 64)  # store 15 steps of history
        obs = self._get_obs(pipeline_state, state_info, obs_history)

        # obs = self._get_obs(pipeline_state, state_info)
        reward, done = jp.zeros(2)
        metrics = {"total_dist": 0.0}
        for k in state_info["rewards"]:
            metrics[k] = state_info["rewards"][k]
        state = State(
            pipeline_state, obs, reward, done, metrics, state_info
        )  # pytype: disable=wrong-arg-types
        return state

    def step(
        self, state: State, action: jax.Array
    ) -> State:  # pytype: disable=signature-mismatch
        rng, cmd_rng, kick_noise_2 = jax.random.split(state.info["rng"], 3)

        # kick
        push_interval = 10
        kick_theta = jax.random.uniform(kick_noise_2, maxval=2 * jp.pi)
        kick = jp.array([jp.cos(kick_theta), jp.sin(kick_theta)])
        kick *= jp.mod(state.info["step"], push_interval) == 0
        qvel = state.pipeline_state.qvel  # pytype: disable=attribute-error
        qvel = qvel.at[:2].set(kick * self._kick_vel + qvel[:2])
        state = state.tree_replace({"pipeline_state.qvel": qvel})

        # physics step
        if self.dial_action_space:
            motor_targets = self.act2joint(action)
        else:
            motor_targets = self._default_pose + action * self._action_scale
        # motor_targets = self._default_pose + action * self._action_scale

        motor_targets = jp.clip(motor_targets, self.lowers, self.uppers)
        pipeline_state = self.pipeline_step(state.pipeline_state, motor_targets)
        x, xd = pipeline_state.x, pipeline_state.xd

        # observation data
        obs = self._get_obs(pipeline_state, state.info, state.obs)
        # obs = self._get_obs(pipeline_state, state.info)
        joint_angles = pipeline_state.q[7:]
        joint_vel = pipeline_state.qd[6:]

        # foot contact data based on z-position
        foot_pos = pipeline_state.site_xpos[
            self._feet_site_id
        ]  # pytype: disable=attribute-error
        foot_contact_z = foot_pos[:, 2] - self._foot_radius
        contact = foot_contact_z < 1e-3  # a mm or less off the floor
        contact_filt_mm = contact | state.info["last_contact"]
        contact_filt_cm = (foot_contact_z < 3e-2) | state.info["last_contact"]
        first_contact = (state.info["feet_air_time"] > 0) * contact_filt_mm
        state.info["feet_air_time"] += self.dt

        ## Head pos
        pos = x.pos[self._torso_idx - 1]
        R = math.quat_to_3x3(x.rot[self._torso_idx - 1])
        head_vec = jnp.array([0.285, 0.0, 0.0])
        head_pos = pos + jnp.dot(R, head_vec)

        # done if joint limits are reached or robot is falling
        up = jp.array([0.0, 0.0, 1.0])
        done = jp.dot(math.rotate(up, x.rot[self._torso_idx - 1]), up) < 0
        done |= jp.any(joint_angles < self.lowers)
        done |= jp.any(joint_angles > self.uppers)
        done |= pipeline_state.x.pos[self._torso_idx - 1, 2] < 0.18

        # if self.deploy:
        #     # done |= jnp.any(pipeline_state.x.pos[:, 2] < 0)
        #     done |= head_pos[2] < 0.

        ## RSL_RL
        # done |= state.info['step'] > 1000

        # reward
        rewards = {
            "tracking_lin_vel": (
                self._reward_tracking_lin_vel(state.info["command"], x, xd)
            ),
            "tracking_ang_vel": (
                self._reward_tracking_ang_vel(state.info["command"], x, xd)
            ),
            "lin_vel_z": self._reward_lin_vel_z(xd),
            "ang_vel_xy": self._reward_ang_vel_xy(xd),
            "orientation": self._reward_orientation(x),
            "torques": self._reward_torques(
                pipeline_state.qfrc_actuator
            ),  # pytype: disable=attribute-error
            "action_rate": self._reward_action_rate(action, state.info["last_act"]),
            "stand_still": self._reward_stand_still(
                state.info["command"],
                joint_angles,
            ),
            "feet_air_time": self._reward_feet_air_time(
                state.info["feet_air_time"],
                first_contact,
                state.info["command"],
            ),
            "foot_slip": self._reward_foot_slip(pipeline_state, contact_filt_cm),
            "termination": self._reward_termination(done, state.info["step"]),
        }

        if self.deploy:
            rewards.update(
                {
                    "gaits": self._reward_gaits(pipeline_state, state.info["step"]),
                    "energy": self._reward_energy(action, pipeline_state),
                    "height": self._reward_height(pipeline_state),
                    "upright": self._reward_upright(pipeline_state),
                    "vel": self._reward_vel(pipeline_state, state.info["command"]),
                    "ang_vel": self._reward_ang_vel(
                        pipeline_state, state.info["command"]
                    ),
                    "yaw": self._reward_yaw(pipeline_state),
                }
            )

        rewards = {
            k: v * self.reward_config.rewards.scales[k] for k, v in rewards.items()
        }

        if self.deploy:
            reward = jp.clip(sum(rewards.values()), -10000.0, 10000.0)
        else:
            reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)

        # RSL_RL
        # reward = jp.clip(sum(rewards.values()), 0.0, 10000.0)

        # state management
        state.info["kick"] = kick
        state.info["last_act"] = action
        state.info["last_ctrl"] = motor_targets
        state.info["last_vel"] = joint_vel
        state.info["feet_air_time"] *= ~contact_filt_mm
        state.info["last_contact"] = contact
        state.info["rewards"] = rewards
        state.info["step"] += 1
        state.info["rng"] = rng

        if self.get_value:
            act_rng, rng = jax.random.split(rng)
            value, _ = self.jit_value_inference_fn(state.obs, act_rng)
            state.info["value"] = value

        # sample new command if more than 500 timesteps achieved
        state.info["command"] = jp.where(
            state.info["step"] > 500,
            self.sample_command(cmd_rng),
            state.info["command"],
        )
        # reset the step counter when done
        state.info["step"] = jp.where(
            done | (state.info["step"] > 500), 0, state.info["step"]
        )

        # log total displacement as a proxy metric
        state.metrics["total_dist"] = math.normalize(x.pos[self._torso_idx - 1])[1]
        state.metrics.update(state.info["rewards"])

        done = jp.float32(done)
        state = state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
        )
        return state

    def step_rsl(self, state: State, action: jax.Array) -> State:
        state = self.step(state, action)
        rng, rng_use = jax.random.split(state.info["rng"], 2)

        state_reset = self.reset(rng_use)

        state.info["rng"] = rng

        state_reset = state_reset.replace(
            done=state.done,
            obs=state.obs,
            reward=state.reward,
        )

        state_ret = jax.lax.cond(
            state.done, lambda x: state_reset, lambda x: state, state.done
        )

        return state_ret

    def _get_obs(
        self,
        pipeline_state: base.State,
        state_info: dict[str, Any],
        obs_history: jax.Array,
    ) -> jax.Array:
        # inv_torso_rot = math.quat_inv(pipeline_state.x.rot[0])
        # local_rpyrate = math.rotate(pipeline_state.xd.ang[0], inv_torso_rot)

        # x, xd = pipeline_state.x, pipeline_state.xd
        # vb = global_to_body_velocity(
        #     xd.vel[self._torso_idx - 1], x.rot[self._torso_idx - 1]
        # )
        # ab = global_to_body_velocity(
        #     xd.ang[self._torso_idx - 1] * jnp.pi / 180.0, x.rot[self._torso_idx - 1]
        # )

        # obs = jp.concatenate([
        #     state_info['command'],  # command
        #     pipeline_state.ctrl,
        #     pipeline_state.qpos,
        #     vb,
        #     ab,
        #     pipeline_state.qvel[6:],
        #     state_info['last_act'],                              # last action
        # ])

        # # clip, noise
        # obs = jp.clip(obs, -100.0, 100.0) + self._obs_noise * jax.random.uniform(
        #     state_info['rng'], obs.shape, minval=-1, maxval=1
        # )
        # # stack observations through time
        # obs = jp.roll(obs_history, obs.size).at[:obs.size].set(obs)

        # inv_torso_rot = math.quat_inv(pipeline_state.x.rot[0])
        # local_rpyrate = math.rotate(pipeline_state.xd.ang[0], inv_torso_rot)

        x, xd = pipeline_state.x, pipeline_state.xd
        vb = global_to_body_velocity(
            xd.vel[self._torso_idx - 1], x.rot[self._torso_idx - 1]
        )
        ab = global_to_body_velocity(
            xd.ang[self._torso_idx - 1] * jnp.pi / 180.0, x.rot[self._torso_idx - 1]
        )

        # obs = jp.concatenate([
        #     jp.array([local_rpyrate[2]]) * 0.25,                 # yaw rate
        #     math.rotate(jp.array([0, 0, -1]), inv_torso_rot),    # projected gravity
        #     state_info['command'] * jp.array([2.0, 2.0, 0.25]),  # command
        #     pipeline_state.q[7:] - self._default_pose,           # motor angles
        #     state_info['last_act'],                              # last action
        # ])

        obs = jp.concatenate(
            [
                state_info["command"],  # command
                pipeline_state.ctrl,
                pipeline_state.qpos,
                vb,
                ab,
                pipeline_state.qvel[6:],
                # state_info['last_act'],                              # last action
                state_info["last_ctrl"],  # last action
            ]
        )

        # clip, noise
        obs = (
            jp.clip(obs, -100.0, 100.0)
            + self._obs_noise
            * jax.random.uniform(state_info["rng"], obs.shape, minval=-1, maxval=1)
            * 0.0
        )
        # stack observations through time
        obs = jp.roll(obs_history, obs.size).at[: obs.size].set(obs)

        return obs

    # ------------ reward functions----------------
    def _reward_lin_vel_z(self, xd: Motion) -> jax.Array:
        # Penalize z axis base linear velocity
        return jp.square(xd.vel[0, 2])

    def _reward_ang_vel_xy(self, xd: Motion) -> jax.Array:
        # Penalize xy axes base angular velocity
        return jp.sum(jp.square(xd.ang[0, :2]))

    def _reward_orientation(self, x: Transform) -> jax.Array:
        # Penalize non flat base orientation
        up = jp.array([0.0, 0.0, 1.0])
        rot_up = math.rotate(up, x.rot[0])
        return jp.sum(jp.square(rot_up[:2]))

    def _reward_torques(self, torques: jax.Array) -> jax.Array:
        # Penalize torques
        return jp.sqrt(jp.sum(jp.square(torques))) + jp.sum(jp.abs(torques))

    def _reward_action_rate(self, act: jax.Array, last_act: jax.Array) -> jax.Array:
        # Penalize changes in actions
        return jp.sum(jp.square(act - last_act))

    def _reward_tracking_lin_vel(
        self, commands: jax.Array, x: Transform, xd: Motion
    ) -> jax.Array:
        # Tracking of linear velocity commands (xy axes)
        local_vel = math.rotate(xd.vel[0], math.quat_inv(x.rot[0]))
        lin_vel_error = jp.sum(jp.square(commands[:2] - local_vel[:2]))
        lin_vel_reward = jp.exp(
            -lin_vel_error / self.reward_config.rewards.tracking_sigma
        )
        return lin_vel_reward

    def _reward_tracking_ang_vel(
        self, commands: jax.Array, x: Transform, xd: Motion
    ) -> jax.Array:
        # Tracking of angular velocity commands (yaw)
        base_ang_vel = math.rotate(xd.ang[0], math.quat_inv(x.rot[0]))
        ang_vel_error = jp.square(commands[2] - base_ang_vel[2])
        return jp.exp(-ang_vel_error / self.reward_config.rewards.tracking_sigma)

    def _reward_feet_air_time(
        self, air_time: jax.Array, first_contact: jax.Array, commands: jax.Array
    ) -> jax.Array:
        # Reward air time.
        rew_air_time = jp.sum((air_time - 0.1) * first_contact)
        rew_air_time *= (
            math.normalize(commands[:2])[1] > 0.05
        )  # no reward for zero command
        return rew_air_time

    def _reward_stand_still(
        self,
        commands: jax.Array,
        joint_angles: jax.Array,
    ) -> jax.Array:
        # Penalize motion at zero commands
        return jp.sum(jp.abs(joint_angles - self._default_pose)) * (
            math.normalize(commands[:2])[1] < 0.1
        )

    def _reward_foot_slip(
        self, pipeline_state: base.State, contact_filt: jax.Array
    ) -> jax.Array:
        # get velocities at feet which are offset from lower legs
        # pytype: disable=attribute-error
        pos = pipeline_state.site_xpos[self._feet_site_id]  # feet position
        feet_offset = pos - pipeline_state.xpos[self._lower_leg_body_id]
        # pytype: enable=attribute-error
        offset = base.Transform.create(pos=feet_offset)
        foot_indices = self._lower_leg_body_id - 1  # we got rid of the world body
        foot_vel = offset.vmap().do(pipeline_state.xd.take(foot_indices)).vel

        # Penalize large feet velocity for feet that are in contact with the ground.
        return jp.sum(jp.square(foot_vel[:, :2]) * contact_filt.reshape((-1, 1)))

    def _reward_termination(self, done: jax.Array, step: jax.Array) -> jax.Array:
        return done & (step < 500)

    def _reward_height(self, pipeline_state: base.State) -> jax.Array:
        x, xd = pipeline_state.x, pipeline_state.xd
        reward_height = -jnp.sum((x.pos[self._torso_idx - 1, 2] - 0.3) ** 2)
        return reward_height

    def _reward_yaw(self, pipeline_state: base.State) -> jax.Array:
        x, xd = pipeline_state.x, pipeline_state.xd
        yaw = math.quat_to_euler(x.rot[self._torso_idx - 1])[2]
        d_yaw = yaw - 0.0
        reward_yaw = -jnp.square(jnp.atan2(jnp.sin(d_yaw), jnp.cos(d_yaw)))
        return reward_yaw

    def _reward_upright(self, pipeline_state: base.State) -> jax.Array:
        x, xd = pipeline_state.x, pipeline_state.xd
        vec_tar = jnp.array([0.0, 0.0, 1.0])
        vec = math.rotate(vec_tar, x.rot[0])
        reward_upright = -jnp.sum(jnp.square(vec - vec_tar))
        return reward_upright

    def _reward_gaits(self, pipeline_state: base.State, steps) -> jax.Array:
        z_feet = pipeline_state.site_xpos[self._feet_site_id][:, 2]
        duty_ratio, cadence, amplitude = self._gait_params[self._gait]
        phases = self._gait_phase[self._gait]
        z_feet_tar = get_foot_step(
            duty_ratio, cadence, amplitude, phases, steps * self.dt
        )
        reward_gaits = -jnp.sum(((z_feet_tar - z_feet) / 0.05) ** 2)

        return reward_gaits

    def _reward_energy(self, ctrl, pipeline_state: base.State) -> jax.Array:
        reward_energy = -jnp.sum(
            jnp.maximum(ctrl * pipeline_state.qvel[6:] / 160.0, 0.0) ** 2
        )
        return reward_energy

    def _reward_vel(self, pipeline_state: base.State, commands: jax.Array) -> jax.Array:
        x, xd = pipeline_state.x, pipeline_state.xd
        vb = global_to_body_velocity(
            xd.vel[self._torso_idx - 1], x.rot[self._torso_idx - 1]
        )
        reward_vel = -jnp.sum((vb[:2] - commands[:2]) ** 2)
        return reward_vel

    def _reward_ang_vel(
        self, pipeline_state: base.State, commands: jax.Array
    ) -> jax.Array:
        x, xd = pipeline_state.x, pipeline_state.xd
        ab = global_to_body_velocity(
            xd.ang[self._torso_idx - 1] * jnp.pi / 180.0, x.rot[self._torso_idx - 1]
        )
        reward_ang_vel = -jnp.sum((ab[2] - commands[2]) ** 2)
        return reward_ang_vel

    def render(
        self,
        trajectory: List[base.State],
        camera: str | None = None,
        width: int = 240,
        height: int = 320,
    ) -> Sequence[np.ndarray]:
        camera = camera or "track"
        return super().render(trajectory, camera=camera, width=width, height=height)


class DialGo2Config(dial_envs.UnitreeGo2EnvConfig):
    leg_control = "position"


class UnitreeGo2DialEnvRL(dial_envs.UnitreeGo2Env):
    def __init__(
        self,
    ):
        config = DialGo2Config()
        super().__init__(config)
        self._action_scale = 0.3

    def _get_obs(
        self,
        pipeline_state: base.State,
        state_info: dict[str, Any],
        obs_history: jax.Array,
    ) -> jax.Array:
        x, xd = pipeline_state.x, pipeline_state.xd
        vb = global_to_body_velocity(
            xd.vel[self._torso_idx - 1], x.rot[self._torso_idx - 1]
        )
        ab = global_to_body_velocity(
            xd.ang[self._torso_idx - 1] * jnp.pi / 180.0, x.rot[self._torso_idx - 1]
        )
        obs = jnp.concatenate(
            [
                state_info["vel_tar"][:2],
                state_info["ang_vel_tar"][2:3],
                pipeline_state.ctrl,
                pipeline_state.qpos,
                vb,
                ab,
                pipeline_state.qvel[6:],
                state_info["last_act"],  # last action
            ]
        )

        # clip, noise
        obs = jp.clip(obs, -100.0, 100.0)
        # stack observations through time
        obs = jp.roll(obs_history, obs.size).at[: obs.size].set(obs)

        return obs

    def reset(self, rng: jax.Array) -> State:  # pytype: disable=signature-mismatch
        rng, key = jax.random.split(rng)

        pipeline_state = self.pipeline_init(self._init_q, jnp.zeros(self._nv))

        state_info = {
            "rng": rng,
            "last_act": jp.zeros(12),
            "pos_tar": jnp.array([0.282, 0.0, 0.3]),
            "vel_tar": jnp.array([0.0, 0.0, 0.0]),
            "ang_vel_tar": jnp.array([0.0, 0.0, 0.0]),
            "yaw_tar": 0.0,
            "step": 0,
            "z_feet": jnp.zeros(4),
            "z_feet_tar": jnp.zeros(4),
            "randomize_target": self._config.randomize_tasks,
            "last_contact": jnp.zeros(4, dtype=jnp.bool),
            "feet_air_time": jnp.zeros(4),
        }

        obs_history = jp.zeros(15 * 31)  # store 15 steps of history
        obs = self._get_obs(pipeline_state, state_info, obs_history)
        reward, done = jnp.zeros(2)
        metrics = {}
        state = State(pipeline_state, obs, reward, done, metrics, state_info)
        return state

    def step(self, state: State, action: jax.Array) -> State:
        rng, cmd_rng = jax.random.split(state.info["rng"], 2)

        # physics step
        # joint_targets = self.act2joint(action)
        joint_targets = self._default_pose + action * self._action_scale
        if self._config.leg_control == "position":
            ctrl = joint_targets
        elif self._config.leg_control == "torque":
            ctrl = self.act2tau(action, state.pipeline_state)
        pipeline_state = self.pipeline_step(state.pipeline_state, ctrl)
        x, xd = pipeline_state.x, pipeline_state.xd

        # observation data
        obs = self._get_obs(pipeline_state, state.info, state.obs)

        # switch to new target if randomize_target is True
        def dont_randomize():
            return (
                jnp.array([self._config.default_vx, self._config.default_vy, 0.0]),
                jnp.array([0.0, 0.0, self._config.default_vyaw]),
            )

        def randomize():
            return self.sample_command(cmd_rng)

        vel_tar, ang_vel_tar = jax.lax.cond(
            (state.info["randomize_target"]) & (state.info["step"] % 500 == 0),
            randomize,
            dont_randomize,
        )
        state.info["vel_tar"] = jnp.minimum(
            vel_tar * state.info["step"] * self.dt / self._config.ramp_up_time, vel_tar
        )
        state.info["ang_vel_tar"] = jnp.minimum(
            ang_vel_tar * state.info["step"] * self.dt / self._config.ramp_up_time,
            ang_vel_tar,
        )

        # reward
        # gaits reward
        z_feet = pipeline_state.site_xpos[self._feet_site_id][:, 2]
        duty_ratio, cadence, amplitude = self._gait_params[self._gait]
        phases = self._gait_phase[self._gait]
        z_feet_tar = get_foot_step(
            duty_ratio, cadence, amplitude, phases, state.info["step"] * self.dt
        )
        reward_gaits = -jnp.sum(((z_feet_tar - z_feet) / 0.05) ** 2)
        # foot contact data based on z-position
        foot_pos = pipeline_state.site_xpos[
            self._feet_site_id
        ]  # pytype: disable=attribute-error
        foot_contact_z = foot_pos[:, 2] - self._foot_radius
        contact = foot_contact_z < 1e-3  # a mm or less off the floor
        contact_filt_mm = contact | state.info["last_contact"]
        contact_filt_cm = (foot_contact_z < 3e-2) | state.info["last_contact"]
        first_contact = (state.info["feet_air_time"] > 0) * contact_filt_mm
        state.info["feet_air_time"] += self.dt
        reward_air_time = jnp.sum((state.info["feet_air_time"] - 0.1) * first_contact)
        # position reward
        pos_tar = (
            state.info["pos_tar"] + state.info["vel_tar"] * self.dt * state.info["step"]
        )
        pos = x.pos[self._torso_idx - 1]
        R = math.quat_to_3x3(x.rot[self._torso_idx - 1])
        head_vec = jnp.array([0.285, 0.0, 0.0])
        head_pos = pos + jnp.dot(R, head_vec)
        reward_pos = -jnp.sum((head_pos - pos_tar) ** 2)
        # stay upright reward
        vec_tar = jnp.array([0.0, 0.0, 1.0])
        vec = math.rotate(vec_tar, x.rot[0])
        reward_upright = -jnp.sum(jnp.square(vec - vec_tar))
        # yaw orientation reward
        yaw_tar = (
            state.info["yaw_tar"]
            + state.info["ang_vel_tar"][2] * self.dt * state.info["step"]
        )
        yaw = math.quat_to_euler(x.rot[self._torso_idx - 1])[2]
        d_yaw = yaw - yaw_tar
        reward_yaw = -jnp.square(jnp.atan2(jnp.sin(d_yaw), jnp.cos(d_yaw)))
        # stay to norminal pose reward
        # reward_pose = -jnp.sum(jnp.square(joint_targets - self._default_pose))
        # velocity reward
        vb = global_to_body_velocity(
            xd.vel[self._torso_idx - 1], x.rot[self._torso_idx - 1]
        )
        ab = global_to_body_velocity(
            xd.ang[self._torso_idx - 1] * jnp.pi / 180.0, x.rot[self._torso_idx - 1]
        )
        reward_vel = -jnp.sum((vb[:2] - state.info["vel_tar"][:2]) ** 2)
        reward_ang_vel = -jnp.sum((ab[2] - state.info["ang_vel_tar"][2]) ** 2)
        # height reward
        reward_height = -jnp.sum(
            (x.pos[self._torso_idx - 1, 2] - state.info["pos_tar"][2]) ** 2
        )
        # energy reward
        reward_energy = -jnp.sum(
            jnp.maximum(ctrl * pipeline_state.qvel[6:] / 160.0, 0.0) ** 2
        )
        # stay alive reward
        reward_alive = 1.0 - state.done
        # reward
        reward = (
            reward_gaits * 0.1
            + reward_air_time * 0.0
            + reward_pos * 0.0
            + reward_upright * 5.0
            + reward_yaw * 1.0
            # + reward_pose * 0.0
            + reward_vel * 1.0
            + reward_ang_vel * 1.0
            + reward_height * 1.0
            + reward_energy * 0.01
            + reward_alive * 0.0
        )

        # done
        up = jnp.array([0.0, 0.0, 1.0])
        joint_angles = pipeline_state.q[7:]
        done = jnp.dot(math.rotate(up, x.rot[self._torso_idx - 1]), up) < 0
        done |= jnp.any(joint_angles < self.joint_range[:, 0])
        done |= jnp.any(joint_angles > self.joint_range[:, 1])
        done |= pipeline_state.x.pos[self._torso_idx - 1, 2] < 0.18
        done = done.astype(jnp.float32)

        # state management
        state.info["step"] += 1
        state.info["rng"] = rng
        state.info["last_act"] = action
        state.info["z_feet"] = z_feet
        state.info["z_feet_tar"] = z_feet_tar
        state.info["feet_air_time"] *= ~contact_filt_mm
        state.info["last_contact"] = contact

        state = state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
        )
        return state


class UnitreeH1EnvRL(PipelineEnv):
    """Environment for training the barkour quadruped joystick policy in MJX."""

    def __init__(
        self,
        obs_noise: float = 0.05,
        action_scale: float = 0.3,
        kick_vel: float = 0.05,
        scene_file: str = "scene_mjx.xml",
        **kwargs,
    ):
        # path = GO2_ROOT_PATH / scene_file
        # sys = mjcf.load(path.as_posix())
        # model_path = get_model_path("unitree_go2", "mjx_scene.xml")
        # model_path = get_model_path("unitree_go2", "mjx_scene_position.xml")
        model_path = get_model_path("unitree_h1", "scene_position_real_feet_lower.xml")
        sys = mjcf.load(model_path)
        self._dt = 0.02  # this environment is 50 fps
        sys = sys.tree_replace({"opt.timestep": 0.02})

        # override menagerie params for smoother policy
        sys = sys.replace(
            dof_damping=sys.dof_damping.at[6:].set(0.5239),
            actuator_gainprm=sys.actuator_gainprm.at[:, 0].set(35.0),
            actuator_biasprm=sys.actuator_biasprm.at[:, 1].set(-35.0),
        )

        n_frames = kwargs.pop("n_frames", int(self._dt / sys.opt.timestep))
        super().__init__(sys, backend="mjx", n_frames=n_frames)

        self.reward_config = get_config()
        # set custom from kwargs
        for k, v in kwargs.items():
            if k.endswith("_scale"):
                self.reward_config.rewards.scales[k[:-6]] = v

        self._torso_idx = mujoco.mj_name2id(
            sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, "torso_link"
        )
        self._action_scale = action_scale
        self._obs_noise = obs_noise
        self._kick_vel = kick_vel
        self._init_q = jp.array(sys.mj_model.keyframe("home").qpos)
        self._default_pose = sys.mj_model.keyframe("home").qpos[7:]
        # self.lowers = jp.array([-0.8, -1.2, -2.5] * 4)
        # self.uppers = jp.array([0.8, 2.3, -0.6] * 4)
        self.lowers = jp.array(
            [
                -0.43,
                -0.43,
                -1.57,
                -0.26,
                -0.87,
                -0.43,
                -0.43,
                -1.57,
                -0.26,
                -0.87,
                -2.35,
                -2.87,
                -0.34,
                -1.3,
                -1.25,
                -2.87,
                -3.11,
                -4.45,
                -1.25,
            ]
        )
        self.uppers = jp.array(
            [
                0.43,
                0.43,
                1.57,
                2.05,
                0.52,
                0.43,
                0.43,
                1.57,
                2.05,
                0.52,
                2.35,
                2.87,
                3.11,
                4.45,
                2.61,
                2.87,
                0.34,
                1.3,
                2.61,
            ]
        )
        self.num_actions = 19
        feet_site = [
            "left_ankle_link",
            "right_ankle_link",
        ]

        feet_site_id = [
            mujoco.mj_name2id(sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, f)
            for f in feet_site
        ]
        assert not any(id_ == -1 for id_ in feet_site_id), "Site not found."
        self._feet_site_id = np.array(feet_site_id)
        lower_leg_body = [
            "left_knee_link",
            "right_knee_link",
        ]
        lower_leg_body_id = [
            mujoco.mj_name2id(sys.mj_model, mujoco.mjtObj.mjOBJ_BODY.value, l)
            for l in lower_leg_body
        ]
        assert not any(id_ == -1 for id_ in lower_leg_body_id), "Body not found."
        self._lower_leg_body_id = np.array(lower_leg_body_id)
        self._foot_radius = 0.0175
        self._nv = sys.nv

    #   @partial(jax.jit, static_argnums=(0,))
    #   def act2joint(self, act: jax.Array) -> jax.Array:
    #       act_normalized = (
    #           act * self._action_scale + 1.0
    #       ) / 2.0  # normalize to [0, 1]
    #       joint_targets = self.lowers + act_normalized * (
    #           self.uppers - self.lowers
    #       )  # scale to joint range
    #       joint_targets = jnp.clip(
    #           joint_targets,
    #           self.lowers,
    #           self.uppers,
    #       )
    #       return joint_targets

    def sample_command(self, rng: jax.Array) -> jax.Array:
        lin_vel_x = [-0.5, 1.0]  # min max [m/s]
        lin_vel_y = [-0.5, 0.5]  # min max [m/s]
        ang_vel_yaw = [-0.5, 0.5]  # min max [rad/s]

        # lin_vel_y = [-0.8, 0.8]  # min max [m/s]
        # lin_vel_y = [-0., 0.]  # min max [m/s]
        # ang_vel_yaw = [-0.7, 0.7]  # min max [rad/s]
        # ang_vel_yaw = [-0., 0.]  # min max [rad/s]

        _, key1, key2, key3 = jax.random.split(rng, 4)
        lin_vel_x = jax.random.uniform(
            key1, (1,), minval=lin_vel_x[0], maxval=lin_vel_x[1]
        )
        lin_vel_y = jax.random.uniform(
            key2, (1,), minval=lin_vel_y[0], maxval=lin_vel_y[1]
        )
        ang_vel_yaw = jax.random.uniform(
            key3, (1,), minval=ang_vel_yaw[0], maxval=ang_vel_yaw[1]
        )
        new_cmd = jp.array([lin_vel_x[0], lin_vel_y[0], ang_vel_yaw[0]])
        return new_cmd

    def reset(self, rng: jax.Array) -> State:  # pytype: disable=signature-mismatch
        rng, key = jax.random.split(rng)

        pipeline_state = self.pipeline_init(self._init_q, jp.zeros(self._nv))

        state_info = {
            "rng": rng,
            "last_act": jp.zeros(19),
            "last_vel": jp.zeros(19),
            "command": self.sample_command(key),
            "last_contact": jp.zeros(2, dtype=bool),
            "feet_air_time": jp.zeros(2),
            "rewards": {k: 0.0 for k in self.reward_config.rewards.scales.keys()},
            "kick": jp.array([0.0, 0.0]),
            "step": 0,
            "vel_tar": jnp.array([0.0, 0.0, 0.0]),
            "ang_vel_tar": jnp.array([0.0, 0.0, 0.0]),
        }
        obs_history = jp.zeros(15 * 92)  # store 15 steps of history
        obs = self._get_obs(pipeline_state, state_info, obs_history)

        # obs = self._get_obs(pipeline_state, state_info)
        reward, done = jp.zeros(2)
        metrics = {"total_dist": 0.0}
        for k in state_info["rewards"]:
            metrics[k] = state_info["rewards"][k]
        state = State(
            pipeline_state, obs, reward, done, metrics, state_info
        )  # pytype: disable=wrong-arg-types
        return state

    def step(
        self, state: State, action: jax.Array
    ) -> State:  # pytype: disable=signature-mismatch
        rng, cmd_rng, kick_noise_2 = jax.random.split(state.info["rng"], 3)

        # kick
        push_interval = 10
        kick_theta = jax.random.uniform(kick_noise_2, maxval=2 * jp.pi)
        kick = jp.array([jp.cos(kick_theta), jp.sin(kick_theta)])
        kick *= jp.mod(state.info["step"], push_interval) == 0
        qvel = state.pipeline_state.qvel  # pytype: disable=attribute-error
        qvel = qvel.at[:2].set(kick * self._kick_vel + qvel[:2])
        state = state.tree_replace({"pipeline_state.qvel": qvel})

        # physics step
        # motor_targets = self.act2joint(action)
        motor_targets = self._default_pose + action * self._action_scale

        motor_targets = jp.clip(motor_targets, self.lowers, self.uppers)
        pipeline_state = self.pipeline_step(state.pipeline_state, motor_targets)
        x, xd = pipeline_state.x, pipeline_state.xd

        # observation data
        obs = self._get_obs(pipeline_state, state.info, state.obs)
        # obs = self._get_obs(pipeline_state, state.info)
        joint_angles = pipeline_state.q[7:]
        joint_vel = pipeline_state.qd[6:]

        # foot contact data based on z-position
        foot_pos = pipeline_state.site_xpos[
            self._feet_site_id
        ]  # pytype: disable=attribute-error
        foot_contact_z = foot_pos[:, 2] - self._foot_radius
        contact = foot_contact_z < 1e-3  # a mm or less off the floor
        contact_filt_mm = contact | state.info["last_contact"]
        contact_filt_cm = (foot_contact_z < 3e-2) | state.info["last_contact"]
        first_contact = (state.info["feet_air_time"] > 0) * contact_filt_mm
        state.info["feet_air_time"] += self.dt

        # done if joint limits are reached or robot is falling
        up = jp.array([0.0, 0.0, 1.0])
        done = jp.dot(math.rotate(up, x.rot[self._torso_idx - 1]), up) < 0
        done |= jp.any(joint_angles < self.lowers)
        done |= jp.any(joint_angles > self.uppers)
        done |= pipeline_state.x.pos[self._torso_idx - 1, 2] < 0.18

        ## RSL_RL
        done |= state.info["step"] > 1000

        # reward
        rewards = {
            "tracking_lin_vel": (
                self._reward_tracking_lin_vel(state.info["command"], x, xd)
            ),
            "tracking_ang_vel": (
                self._reward_tracking_ang_vel(state.info["command"], x, xd)
            ),
            "lin_vel_z": self._reward_lin_vel_z(xd),
            "ang_vel_xy": self._reward_ang_vel_xy(xd),
            "orientation": self._reward_orientation(x),
            "torques": self._reward_torques(
                pipeline_state.qfrc_actuator
            ),  # pytype: disable=attribute-error
            "action_rate": self._reward_action_rate(action, state.info["last_act"]),
            "stand_still": self._reward_stand_still(
                state.info["command"],
                joint_angles,
            ),
            "feet_air_time": self._reward_feet_air_time(
                state.info["feet_air_time"],
                first_contact,
                state.info["command"],
            ),
            "foot_slip": self._reward_foot_slip(pipeline_state, contact_filt_cm),
            "termination": self._reward_termination(done, state.info["step"]),
        }
        rewards = {
            k: v * self.reward_config.rewards.scales[k] for k, v in rewards.items()
        }

        # reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)

        # RSL_RL
        reward = jp.clip(sum(rewards.values()), 0.0, 10000.0)

        # state management
        state.info["kick"] = kick
        state.info["last_act"] = action
        state.info["last_vel"] = joint_vel
        state.info["feet_air_time"] *= ~contact_filt_mm
        state.info["last_contact"] = contact
        state.info["rewards"] = rewards
        state.info["step"] += 1
        state.info["rng"] = rng

        # sample new command if more than 500 timesteps achieved
        state.info["command"] = jp.where(
            state.info["step"] > 500,
            self.sample_command(cmd_rng),
            state.info["command"],
        )
        # reset the step counter when done
        state.info["step"] = jp.where(
            done | (state.info["step"] > 500), 0, state.info["step"]
        )

        # log total displacement as a proxy metric
        state.metrics["total_dist"] = math.normalize(x.pos[self._torso_idx - 1])[1]
        state.metrics.update(state.info["rewards"])

        done = jp.float32(done)
        state = state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
        )
        return state

    def step_rsl(self, state: State, action: jax.Array) -> State:
        state = self.step(state, action)
        rng, rng_use = jax.random.split(state.info["rng"], 2)

        state_reset = self.reset(rng_use)

        state.info["rng"] = rng

        state_reset = state_reset.replace(
            done=state.done,
            obs=state.obs,
            reward=state.reward,
        )

        state_ret = jax.lax.cond(
            state.done, lambda x: state_reset, lambda x: state, state.done
        )

        return state_ret

    def _get_obs(
        self,
        pipeline_state: base.State,
        state_info: dict[str, Any],
        obs_history: jax.Array,
    ) -> jax.Array:
        # inv_torso_rot = math.quat_inv(pipeline_state.x.rot[0])
        # local_rpyrate = math.rotate(pipeline_state.xd.ang[0], inv_torso_rot)

        # x, xd = pipeline_state.x, pipeline_state.xd
        # vb = global_to_body_velocity(
        #     xd.vel[self._torso_idx - 1], x.rot[self._torso_idx - 1]
        # )
        # ab = global_to_body_velocity(
        #     xd.ang[self._torso_idx - 1] * jnp.pi / 180.0, x.rot[self._torso_idx - 1]
        # )

        # obs = jp.concatenate([
        #     state_info['command'],  # command
        #     pipeline_state.ctrl,
        #     pipeline_state.qpos,
        #     vb,
        #     ab,
        #     pipeline_state.qvel[6:],
        #     state_info['last_act'],                              # last action
        # ])

        # # clip, noise
        # obs = jp.clip(obs, -100.0, 100.0) + self._obs_noise * jax.random.uniform(
        #     state_info['rng'], obs.shape, minval=-1, maxval=1
        # )
        # # stack observations through time
        # obs = jp.roll(obs_history, obs.size).at[:obs.size].set(obs)

        # inv_torso_rot = math.quat_inv(pipeline_state.x.rot[0])
        # local_rpyrate = math.rotate(pipeline_state.xd.ang[0], inv_torso_rot)

        x, xd = pipeline_state.x, pipeline_state.xd
        vb = global_to_body_velocity(
            xd.vel[self._torso_idx - 1], x.rot[self._torso_idx - 1]
        )
        ab = global_to_body_velocity(
            xd.ang[self._torso_idx - 1] * jnp.pi / 180.0, x.rot[self._torso_idx - 1]
        )

        # obs = jp.concatenate([
        #     jp.array([local_rpyrate[2]]) * 0.25,                 # yaw rate
        #     math.rotate(jp.array([0, 0, -1]), inv_torso_rot),    # projected gravity
        #     state_info['command'] * jp.array([2.0, 2.0, 0.25]),  # command
        #     pipeline_state.q[7:] - self._default_pose,           # motor angles
        #     state_info['last_act'],                              # last action
        # ])

        obs = jp.concatenate(
            [
                state_info["command"],  # command
                pipeline_state.ctrl,
                pipeline_state.qpos,
                vb,
                ab,
                pipeline_state.qvel[6:],
                state_info["last_act"],  # last action
            ]
        )

        # print(obs.shape)

        # clip, noise
        obs = (
            jp.clip(obs, -100.0, 100.0)
            + self._obs_noise
            * jax.random.uniform(state_info["rng"], obs.shape, minval=-1, maxval=1)
            * 0.0
        )
        # stack observations through time
        obs = jp.roll(obs_history, obs.size).at[: obs.size].set(obs)

        return obs

    # ------------ reward functions----------------
    def _reward_lin_vel_z(self, xd: Motion) -> jax.Array:
        # Penalize z axis base linear velocity
        return jp.square(xd.vel[0, 2])

    def _reward_ang_vel_xy(self, xd: Motion) -> jax.Array:
        # Penalize xy axes base angular velocity
        return jp.sum(jp.square(xd.ang[0, :2]))

    def _reward_orientation(self, x: Transform) -> jax.Array:
        # Penalize non flat base orientation
        up = jp.array([0.0, 0.0, 1.0])
        rot_up = math.rotate(up, x.rot[0])
        return jp.sum(jp.square(rot_up[:2]))

    def _reward_torques(self, torques: jax.Array) -> jax.Array:
        # Penalize torques
        return jp.sqrt(jp.sum(jp.square(torques))) + jp.sum(jp.abs(torques))

    def _reward_action_rate(self, act: jax.Array, last_act: jax.Array) -> jax.Array:
        # Penalize changes in actions
        return jp.sum(jp.square(act - last_act))

    def _reward_tracking_lin_vel(
        self, commands: jax.Array, x: Transform, xd: Motion
    ) -> jax.Array:
        # Tracking of linear velocity commands (xy axes)
        local_vel = math.rotate(xd.vel[0], math.quat_inv(x.rot[0]))
        lin_vel_error = jp.sum(jp.square(commands[:2] - local_vel[:2]))
        lin_vel_reward = jp.exp(
            -lin_vel_error / self.reward_config.rewards.tracking_sigma
        )
        return lin_vel_reward

    def _reward_tracking_ang_vel(
        self, commands: jax.Array, x: Transform, xd: Motion
    ) -> jax.Array:
        # Tracking of angular velocity commands (yaw)
        base_ang_vel = math.rotate(xd.ang[0], math.quat_inv(x.rot[0]))
        ang_vel_error = jp.square(commands[2] - base_ang_vel[2])
        return jp.exp(-ang_vel_error / self.reward_config.rewards.tracking_sigma)

    def _reward_feet_air_time(
        self, air_time: jax.Array, first_contact: jax.Array, commands: jax.Array
    ) -> jax.Array:
        # Reward air time.
        rew_air_time = jp.sum((air_time - 0.1) * first_contact)
        rew_air_time *= (
            math.normalize(commands[:2])[1] > 0.05
        )  # no reward for zero command
        return rew_air_time

    def _reward_stand_still(
        self,
        commands: jax.Array,
        joint_angles: jax.Array,
    ) -> jax.Array:
        # Penalize motion at zero commands
        return jp.sum(jp.abs(joint_angles - self._default_pose)) * (
            math.normalize(commands[:2])[1] < 0.1
        )

    def _reward_foot_slip(
        self, pipeline_state: base.State, contact_filt: jax.Array
    ) -> jax.Array:
        # get velocities at feet which are offset from lower legs
        # pytype: disable=attribute-error
        pos = pipeline_state.site_xpos[self._feet_site_id]  # feet position
        feet_offset = pos - pipeline_state.xpos[self._lower_leg_body_id]
        # pytype: enable=attribute-error
        offset = base.Transform.create(pos=feet_offset)
        foot_indices = self._lower_leg_body_id - 1  # we got rid of the world body
        foot_vel = offset.vmap().do(pipeline_state.xd.take(foot_indices)).vel

        # Penalize large feet velocity for feet that are in contact with the ground.
        return jp.sum(jp.square(foot_vel[:, :2]) * contact_filt.reshape((-1, 1)))

    def _reward_termination(self, done: jax.Array, step: jax.Array) -> jax.Array:
        return done & (step < 500)

    def render(
        self,
        trajectory: List[base.State],
        camera: str | None = None,
        width: int = 240,
        height: int = 320,
    ) -> Sequence[np.ndarray]:
        camera = camera or "track"
        return super().render(trajectory, camera=camera, width=width, height=height)
