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
# import torch

from sac_mppi.sac import SACMPPIEnv
from sac_mppi import RL_LOG_DIR
import dial_mpc.envs as dial_envs
from dial_mpc.core.dial_config import DialConfig
from dial_mpc.utils.io_utils import get_example_path, load_dataclass_from_dict
from dial_mpc.envs import UnitreeGo2EnvConfig
from rsl_rl.runners import OnPolicyRunner
def train():
    # config_dict = yaml.safe_load(open("/home/wenli/SAC-MPPI/sac_mppi/dial_mpc/dial_mpc/examples/unitree_go2_trot.yaml"))
    log_root = os.path.join(RL_LOG_DIR, "test")
    log_dir = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + "test")

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
        # 'leg_control': "position",
        'action_scale': 1.0,
        'backend': 'mjx',
    }
    config_dict = UnitreeGo2EnvConfig()
    num_envs=1024
    # num_envs=2
    env = SACMPPIEnv(config_dict, num_envs)
    
    # ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    # ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)
    train_config = yaml.safe_load(open("/home/wenli/SAC-MPPI/sac_mppi/rsl_rl/config/dummy_config.yaml"))
    ppo_runner = OnPolicyRunner(env, train_config, log_dir, device="cuda:0")
    
    # ppo_runner.load("/home/wenli/SAC-MPPI/sac_mppi/logs/test/Nov29_21-37-12_test/model_600.pt")
    
    ppo_runner.learn(num_learning_iterations=train_config['runner']['max_iterations'], init_at_random_ep_len=True)
    
if __name__ == '__main__':
    # args = get_args()
    # train(args)
    train()