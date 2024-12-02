#!/usr/bin/env python3
# @title Import packages for plotting and creating graphics
import itertools
import time
from typing import Callable
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Union

import matplotlib.pyplot as plt
import numpy as np

# More legible printing from numpy.
np.set_printoptions(precision=3, suppress=True, linewidth=100)

# @title Import MuJoCo, MJX, and Brax
from datetime import datetime
import functools
import os


import jax
import numpy as np
from flax.training import orbax_utils
from flax import struct
from matplotlib import pyplot as plt
import mediapy as media
from orbax import checkpoint as ocp

import mujoco
from mujoco import mjx

from brax import base
from brax import envs
from brax import math
from brax.base import Base, Motion, Transform
from brax.base import State as PipelineState
from brax.envs.base import Env, PipelineEnv, State
from brax.mjx.base import State as MjxState
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import html, mjcf, model

from sac_mppi import RL_LOG_DIR, RSL_RL_ROOT_DIR

from sac_mppi.dial_mpc.dial_mpc.envs.unitree_go2_env import UnitreeGo2Env
from sac_mppi.dial_mpc.dial_mpc.envs.unitree_go2_env import UnitreeGo2EnvConfig
from sac_mppi.dial_mpc.dial_mpc.utils.function_utils import (
    global_to_body_velocity,
    get_foot_step,
)
import jax.numpy as jnp
from dial_mpc.utils.io_utils import get_model_path
from brax.base import System
from sac_mppi.brax_rl.brax_env import UnitreeGo2EnvRL, UnitreeH1EnvRL
from sac_mppi.brax_rl.brax_utils import train

log_root = os.path.join(RL_LOG_DIR, "brax_go2")
log_dir = os.path.join(
    log_root, datetime.now().strftime("%b%d_%H-%M-%S") + "_" + "walk"
)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

env = UnitreeGo2EnvRL()
# env = UnitreeH1EnvRL()
sys = env.sys
# define the jit reset/step functions
jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)
# initialize the state
state = jit_reset(jax.random.PRNGKey(0))
rollout = [state.pipeline_state]

make_networks_factory = functools.partial(
    ppo_networks.make_ppo_networks, policy_hidden_layer_sizes=(512, 256, 128)
)

train_fn = functools.partial(
    train,
    num_timesteps=100_000_000,
    num_evals=10,
    reward_scaling=1.0,
    episode_length=1000,
    normalize_observations=True,
    action_repeat=1,
    unroll_length=20,
    num_minibatches=32,
    num_updates_per_batch=4,
    discounting=0.97,
    learning_rate=3e-4,
    entropy_cost=1e-2,
    num_envs=8192,
    batch_size=256,
    seed=0,
    network_factory=make_networks_factory,
    # randomization_fn=domain_randomize,
)


x_data = []
y_data = []
ydataerr = []
times = [datetime.now()]

max_y, min_y = 13000, 0


def progress(num_steps, metrics):
    times.append(datetime.now())
    x_data.append(num_steps)
    y_data.append(metrics["eval/episode_reward"])
    ydataerr.append(metrics["eval/episode_reward_std"])

    print(
        f'step: {num_steps}, reward: {metrics["eval/episode_reward"]:.3f}, time: {times[-1] - times[0]}'
    )


#   plt.xlim([0, train_fn.keywords['num_timesteps'] * 1.25])
#   # plt.ylim([min_y, max_y])

#   plt.xlabel('# environment steps')
#   plt.ylabel('reward per episode')
#   plt.title(f'y={y_data[-1]:.3f}')

#   plt.errorbar(
#       x_data, y_data, yerr=ydataerr)
#   plt.show()

make_inference_fn, params, value_params, _ = train_fn(
    environment=env, progress_fn=progress
)

print(f"time to jit: {times[1] - times[0]}")
print(f"time to train: {times[-1] - times[1]}")


# @title Save Model
policy_model_path = os.path.join(log_dir, "go2_policy")
value_model_path = os.path.join(log_dir, "go2_value")
model.save_params(policy_model_path, params)
model.save_params(value_model_path, value_params)

print(f"Saved policy model to {policy_model_path}")
print(f"Saved value model to {value_model_path}")
