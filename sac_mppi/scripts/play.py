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

from sac_mppi.sac import SACMPPIEnv
import dial_mpc.envs as dial_envs
from dial_mpc.core.dial_config import DialConfig
from dial_mpc.utils.io_utils import get_example_path, load_dataclass_from_dict
from dial_mpc.envs import UnitreeGo2EnvConfig
from rsl_rl.runners import OnPolicyRunner
from brax.io import html
import jax.numpy as jnp
from sac_mppi import RL_LOG_DIR

def train():
    # config_dict = yaml.safe_load(open("/home/wenli/SAC-MPPI/sac_mppi/dial_mpc/dial_mpc/examples/unitree_go2_trot.yaml"))

    config_dict = {
        'task_name': "test",
        'randomize_tasks': False,
        'kp': 30.0,
        'kd': 1.,
        'debug': False,
        'dt': 0.02,
        'timestep': 0.005,
        'backend': "mjx",
        'leg_control': "torque",
        # "leg_control": "position",
        'action_scale': 1.0,
        'backend': 'mjx',
    }
    config_dict = UnitreeGo2EnvConfig()
    num_envs=1
    env = SACMPPIEnv(config_dict, num_envs)
    
    rollout = []
    
    # joint_range [[-0.5   0.5 ]
    # [ 0.4   1.4 ]
    # [-2.3  -0.85]
    # [-0.5   0.5 ]
    # [ 0.4   1.4 ]
    # [-2.3  -0.85]
    # [-0.5   0.5 ]
    # [ 0.4   1.4 ]
    # [-2.3  -1.3 ]
    # [-0.5   0.5 ]
    # [ 0.4   1.4 ]
    # [-2.3  -1.3 ]]
    train_config = yaml.safe_load(open("/home/wenli/SAC-MPPI/sac_mppi/rsl_rl/config/dummy_config.yaml"))
    log_root = os.path.join(RL_LOG_DIR, "test")
    log_dir = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + "test")
    ppo_runner = OnPolicyRunner(env, train_config, log_dir, device="cuda:0")
    ppo_runner.load("/home/wenli/SAC-MPPI/sac_mppi/logs/test/Nov29_23-28-14_test/model_2700.pt")
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    obs, _ = env.reset()
    for i in range(1000):
        pipeline_state = env.states.pipeline_state
        ps_x = pipeline_state.x
        ps_x = ps_x.replace(
            pos = ps_x.pos[0],
            rot = ps_x.rot[0],
        )
        pipeline_state = pipeline_state.replace(
            x=ps_x,
        )
        rollout.append(pipeline_state)
        # actions = torch.zeros((num_envs, env.num_actions))
        actions = policy(obs) 
        # actions = torch.rand_like(actions) * 2 - 1
        # actions = actions * 0.
        print(actions)
        obs, rews, dones, infos = env.step(actions)
        # joint_angles = pipeline_state.q[:, 7:]
        # print(joint_angles > env.joint_range[:, 1])
        # print(joint_angles < env.joint_range[:, 0])
        print("outer step", i)
        print("inner step", env.states.info['step'])
        print( "dones", dones)
        # print("reward_vel", env.states.info['reward_vel'])
        print("head_pos", env.states.info['head_pos'])
        print("pos", pipeline_state.x.pos[2])
        print("vel_tar", env.states.info['vel_tar'])
        print("")
    
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
                    pipeline_state.qpos[0],
                    pipeline_state.qvel[0],
                    pipeline_state.ctrl[0],
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