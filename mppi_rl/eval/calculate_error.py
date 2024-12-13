import argparse

import jax
from jax import numpy as jnp

parser = argparse.ArgumentParser(description="Process observation directory.")
parser.add_argument(
    "--observation_dir", type=str, required=True, help="Path to the observations file."
)
args = parser.parse_args()

# Load observations
# observations_dir = "/home/zhixuan/16831/project/MPPI-RL/unitree_h1_jog/20241202-192906_observations.npy"
observations_dir = args.observation_dir
observations = jnp.load(observations_dir)

nu_dict = {
    "unitree_h1_jog": 19,
    "unitree_h1_loco": 11,
    "unitree_h1_push_crate": 19,
    "unitree_go2_trot": 12,
    "unitree_go2_seq_jump": 12,
    "unitree_go2_crate_climb": 12,
}
Nq_dict = {
    "unitree_h1_jog": 26,
    "unitree_h1_loco": 18,
    "unitree_h1_push_crate": 27,
    "unitree_go2_trot": 19,
    "unitree_go2_seq_jump": 18,
    "unitree_go2_crate_climb": 19,
}

found_key = None
for key in nu_dict.keys():
    if key in observations_dir:
        found_key = key
        break

if found_key is not None:
    nu = nu_dict[found_key]
    Nq = Nq_dict[found_key]

# Compute indices
vb_start = 6 + nu + Nq  # 6 (vel_tar and ang_vel_tar) + nu + Nq
vb_end = vb_start + 3  # vb has 3 elements
ab_start = vb_end  # ab starts immediately after vb
ab_end = ab_start + 3  # ab has 3 elements

# Extract ground truth commands
vel_tar = observations[:, :3]
ang_vel_tar = observations[:, 3:6]

# Extract actual velocities
vb = observations[:, vb_start:vb_end]
ab = observations[:, ab_start:ab_end]

# Compute tracking errors
tracking_error_linear = jnp.linalg.norm(vb[:, :2] - vel_tar[:, :2], axis=1)
tracking_error_angular = jnp.abs(ab[:, 2] - ang_vel_tar[:, 2])

# Report the errors
mean_tracking_error_linear = jnp.mean(tracking_error_linear)
mean_tracking_error_angular = jnp.mean(tracking_error_angular)
print(f"Mean Linear Tracking Error: {mean_tracking_error_linear}")
print(f"Mean Angular Tracking Error: {mean_tracking_error_angular}")
