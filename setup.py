from setuptools import setup, find_packages

setup(
    name="hires_vic",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "gymnasium",
        "stable-baselines3",
        "robosuite",
        "torch",
        "numpy"
    ],
)