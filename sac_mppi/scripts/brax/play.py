#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin
import functools
import os
from datetime import datetime

import jax
import jax.numpy as jnp
import numpy as np
import torch
import yaml
from brax.io import html
from brax.io import mjcf
from brax.io import model
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo

from sac_mppi import RL_LOG_DIR
from sac_mppi.brax_rl import brax_utils
from sac_mppi.brax_rl.brax_env import UnitreeGo2DialEnvRL
from sac_mppi.brax_rl.brax_env import UnitreeGo2EnvRL
from sac_mppi.brax_rl.brax_env import UnitreeH1EnvRL

# import isaacgym
# from legged_gym.envs import *
# from legged_gym.utils import get_args, task_registry

visualization_dir = os.path.join(RL_LOG_DIR, "visualization")

if not os.path.exists(visualization_dir):
    os.makedirs(visualization_dir)


def train():

    env = UnitreeGo2EnvRL(deploy=True)
    # env = UnitreeGo2DialEnvRL()
    # env = UnitreeH1EnvRL()
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    rng = jax.random.PRNGKey(0)
    state = jit_reset(rng)

    rollout = []

    print("env")

    make_networks_factory = functools.partial(
        ppo_networks.make_ppo_networks, policy_hidden_layer_sizes=(512, 256, 128)
    )

    print(f"obs shape: {state.obs.shape}, action size: {env.action_size}")
    ppo_network = make_networks_factory(
        state.obs.shape[-1],
        env.action_size,
        preprocess_observations_fn=running_statistics.normalize,
    )
    make_inference_fn = brax_utils.make_inference_fn(ppo_network)
    make_value_inference_fn = brax_utils.make_value_inference_fn(ppo_network)

    model_path = (
        "/home/wenli/SAC-MPPI/sac_mppi/logs/brax_go2/ppo/Dec10_13-18-53_walk/policy_step100270080"
    )
    value_model_path = (
        "/home/wenli/SAC-MPPI/sac_mppi/logs/brax_go2/Dec02_14-52-19_walk/go2_value"
    )

    params = model.load_params(model_path)
    value_params = model.load_params(value_model_path)
    inference_fn = make_inference_fn(params)
    value_inference_fn = make_value_inference_fn(value_params)
    jit_inference_fn = jax.jit(inference_fn)
    jit_value_inference_fn = jax.jit(value_inference_fn)

    for i in range(1000):
        pipeline_state = state.pipeline_state
        rollout.append(pipeline_state)

        act_rng, rng = jax.random.split(rng)
        ctrl, _ = jit_inference_fn(state.obs, act_rng)

        act_rng, rng = jax.random.split(rng)
        value, _ = jit_value_inference_fn(state.obs, act_rng)
        print("value", value)

        state = jit_step(state, ctrl)
        # print("outer step", i)
        # print("inner step", state.info['step'])
        # if state.done:
        #     act_rng, rng = jax.random.split(rng, 2)
        #     state = jit_reset(act_rng)

    print("Processing rollout for visualization")
    import flask

    app = flask.Flask(__name__)
    webpage = html.render(
        env.sys.tree_replace({"opt.timestep": env.dt}), rollout, 1080, True
    )

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    # save the html file
    with open(
        os.path.join(
            RL_LOG_DIR, "visualization", f"{timestamp}_brax_visualization.html"
        ),
        "w",
    ) as f:
        f.write(webpage)

    # save the rollout
    data = []
    # xdata = []
    for i in range(len(rollout)):
        pipeline_state = rollout[i]
        data.append(
            jnp.concatenate(
                [
                    jnp.array([i]),
                    pipeline_state.qpos,
                    pipeline_state.qvel,
                    pipeline_state.ctrl,
                ]
            )
        )
        # xdata.append(infos[i]["xbar"][-1])
    data = jnp.array(data)
    # xdata = jnp.array(xdata)
    jnp.save(os.path.join(RL_LOG_DIR, "visualization", f"{timestamp}_states"), data)

    @app.route("/")
    def index():
        return webpage

    app.run(port=5000)


if __name__ == "__main__":
    # args = get_args()
    # train(args)
    train()
