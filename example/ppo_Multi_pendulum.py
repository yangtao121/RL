import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from common.args import EnvArgs, HyperParameter
from algo.PPO_Multi import PPO_Multi

from common.policy import Gaussian_policy
from common.critic import Critic
from common.NeuralNet import gaussian_mlp, mlp

worker_num = 5
env_name = "Pendulum-v0"
observation_dims = 3
action_dims = 1

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

policy = Gaussian_policy(Model=actor)
critic = Critic(Model=value_net)

ppo = PPO_Multi(
    policy=policy,
    critic=critic,
    env_args=env_args,
    hyper_parameter=hyper_parameters,
    net_visualize=True
)

ppo.train(env_name)
