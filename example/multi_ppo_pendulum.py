import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import gym
from common.args import EnvArgs, HyperParameter
from algo.PPO import PPO

from common.worker import MultiWorker
from common.policy import Gaussian_policy
from common.critic import Critic
from common.NeuralNet import gaussian_mlp, mlp

# import multiprocessing as mp
#
# mp.set_start_method('forkserver')


worker_num = 10
env = gym.make("Pendulum-v0").unwrapped
observation_dims = env.observation_space.shape[0]
action_dims = env.action_space.shape[0]
envs = [gym.make("Pendulum-v0").unwrapped for i in range(worker_num)]

env_args = EnvArgs(
    trajs=2,
    steps=200,
    epochs=120,
    batch_size=40,
    observation_dims=observation_dims,
    action_dims=action_dims,
    multi_worker_num=worker_num
)

hyper_parameters = HyperParameter(
    clip_ratio=0.2,
    policy_learning_rate=3e-4,
    critic_learning_rate=1e-3,
    update_steps=10,
    gamma=0.99,
    lambada=0.95
)

actor = gaussian_mlp(
    state_dims=env_args.observation_dims,
    output_dims=env_args.action_dims,
    hidden_size=(64, 64),
    name='actor'
)

value_net = mlp(
    state_dims=env_args.observation_dims,
    output_dims=1,
    hidden_size=(32, 32),
    name='value'
)

worker = MultiWorker(
    envs=envs,
    env_args=env_args,
    hyper_parameter=hyper_parameters
)

policy = Gaussian_policy(Model=actor)
critic = Critic(Model=value_net)

ppo = PPO(
    policy=policy,
    critic=critic,
    worker=worker,
    env_args=env_args,
    hyper_parameter=hyper_parameters,
    net_visualize=True
)

ppo.train()
