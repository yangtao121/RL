from setuptools import setup

setup(
    name="RL",
    version="1.0",
    author="Tao Yang",
    author_email="291843078@qq.com",
    py_modules=[
        "RL.common.args",
        "RL.common.collector",
        "RL.common.critic",
        "RL.common.functions",
        "RL.common.NeuralNet",
        "RL.common.policy",
        "RL.common.worker",
        "RL.algo.PPO",
    ]
)
