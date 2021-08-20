from setuptools import setup

setup(
    name="RL",
    version="1.5",
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
    ],
    install_requires=['tensorflow>=2.6.0', 'tensorflow-probability>=0.13']
)
