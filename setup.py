
from setuptools import find_packages, setup

setup(
    name="sac_mppi",
    version="2.0.2",
    packages=find_packages(),
    author="ETH Zurich, NVIDIA CORPORATION",
    maintainer="Nikita Rudin, David Hoeller",
    maintainer_email="rudinn@ethz.ch",
    url="https://github.com/leggedrobotics/rsl_rl",
    license="BSD-3",
    description="Fast and simple RL algorithms implemented in pytorch",
    python_requires=">=3.10",
    install_requires=[
        
    ],
)
