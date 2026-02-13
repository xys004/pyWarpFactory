from setuptools import setup, find_packages

setup(
    name="StaticWarpBubbles",
    version="0.1.0",
    description="2025 Static Warp Bubble extensions for WarpFactory",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
    ],
    # Assumes warpfactory is installed in the environment
)
