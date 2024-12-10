#!/usr/bin/env python3
# @title Import packages for plotting and creating graphics
import functools
import itertools
import os
import time
from datetime import datetime
from typing import Callable
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Union

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mediapy as media
import mujoco
import numpy as np
from brax import base
from brax import envs
from brax import math
from brax.base import Base
from brax.base import Motion
from brax.base import State as PipelineState
from brax.base import System
from brax.base import Transform
from brax.envs.base import Env
from brax.envs.base import PipelineEnv
from brax.envs.base import State
from brax.io import html
from brax.io import mjcf
from brax.io import model
from brax.mjx.base import State as MjxState
from brax.training.agents.sac import train as sac
from brax.training.agents.sac import networks as sac_networks
from dial_mpc.utils.io_utils import get_model_path
from flax import struct
from flax.training import orbax_utils
from matplotlib import pyplot as plt
from mujoco import mjx
from orbax import checkpoint as ocp

from sac_mppi import RL_LOG_DIR
from sac_mppi import RSL_RL_ROOT_DIR
from sac_mppi.brax_rl.brax_env import UnitreeGo2EnvRL
from sac_mppi.brax_rl.brax_env import UnitreeH1EnvRL
from sac_mppi.brax_rl.brax_utils import train
from sac_mppi.dial_mpc.dial_mpc.envs.unitree_go2_env import UnitreeGo2Env
from sac_mppi.dial_mpc.dial_mpc.envs.unitree_go2_env import UnitreeGo2EnvConfig
from sac_mppi.dial_mpc.dial_mpc.utils.function_utils import get_foot_step
from sac_mppi.dial_mpc.dial_mpc.utils.function_utils import global_to_body_velocity

# @title Import MuJoCo, MJX, and Brax


# More legible printing from numpy.
np.set_printoptions(precision=3, suppress=True, linewidth=100)

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

#debug
for i in range(10):
    ctrl = 0.0 * jnp.ones(env.sys.nu)
    state = jit_step(state, ctrl)
    rollout.append(state.pipeline_state)
import pdb;pdb.set_trace()

make_networks_factory = functools.partial(
    sac_networks.make_sac_networks, 
    hidden_layer_sizes = (512, 256, 128),
    # activation = linen.relu,
    # policy_network_layer_norm = False,
    # q_network_layer_norm = False,
)

train_fn = functools.partial(
    sac.train,
    num_timesteps=10_000_000,             # Reduced from PPO to balance training time
    num_evals=10,                         
    reward_scaling=1.0,                   
    episode_length=1000,                  
    normalize_observations=True,          
    action_repeat=1,                      
    num_envs=128,                         # Fewer parallel environments
    num_eval_envs=128,                    
    learning_rate=1e-4,                   # Smaller lr
    discounting=0.99,                     # Higher discount factor
    batch_size=256,                       
    seed=0,                               
    tau=0.005,                            # Target network update rate
    min_replay_size=10_000,               # Minimum replay buffer size before training
    max_replay_size=1_000_000,            # Maximum replay buffer size
    grad_updates_per_step=1,              
    # deterministic_eval=False,             # Non-deterministic evaluation for SAC
    network_factory=make_networks_factory, # Factory for SAC networks
    randomization_fn=None,                # Placeholder for domain randomization
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
