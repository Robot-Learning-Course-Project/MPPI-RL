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

import numpy as np
import os
from datetime import datetime
import yaml

# import isaacgym
# from legged_gym.envs import *
# from legged_gym.utils import get_args, task_registry
import torch

from sac_mppi.brax_rl.brax_env import UnitreeH1EnvRL
import dial_mpc.envs as dial_envs
from dial_mpc.core.dial_config import DialConfig
from dial_mpc.utils.io_utils import get_example_path, load_dataclass_from_dict
from brax.io import html
import jax.numpy as jnp
import jax


def train():
    # config_dict = yaml.safe_load(open("/home/wenli/SAC-MPPI/sac_mppi/dial_mpc/dial_mpc/examples/unitree_go2_trot.yaml"))

    
    env = UnitreeH1EnvRL()
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    rng = jax.random.PRNGKey(0)
    state = jit_reset(rng)
    
    rollout = []
    
    zero_actions = jnp.zeros((env.num_actions))
    
    for i in range(1000):
        pipeline_state = state.pipeline_state
        rollout.append(pipeline_state)
        act_rng, rng = jax.random.split(rng)

        state = jit_step(state, zero_actions)
        print("outer step", i)
        print("inner step", state.info['step'])
        if state.done:
            act_rng, rng = jax.random.split(rng, 2)
            state = jit_reset(act_rng)
    
    print("Processing rollout for visualization")
    import flask

    app = flask.Flask(__name__)
    webpage = html.render(
        env.sys.tree_replace({"opt.timestep": env.dt}), rollout, 1080, True
    )

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    # save the html file
    with open(
        os.path.join("/home/wenli/SAC-MPPI/sac_mppi/test", f"{timestamp}_brax_visualization.html"),
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
    jnp.save(os.path.join("/home/wenli/SAC-MPPI/sac_mppi/test", f"{timestamp}_states"), data)
    # jnp.save(os.path.join("/home/wenli/SAC-MPPI/sac_mppi/test", f"{timestamp}_predictions"), xdata)

    @app.route("/")
    def index():
        return webpage

    app.run(port=5000)
    
if __name__ == '__main__':
    # args = get_args()
    # train(args)
    train()