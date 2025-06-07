from setuptools import setup, find_packages

setup(
    name="actor_critic_methods",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "gymnasium",
        "numpy",
        "pygame",
        "torch",
        "matplotlib",
    ],
)
