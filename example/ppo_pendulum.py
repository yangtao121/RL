import gym
from common.args import EnvArgs, HyperParameter
from algo.PPO import PPO

from common.worker import Worker
from common.policy import Gaussian_policy
from common.critic import Critic
from common.NeuralNet import gaussian_mlp, mlp

env = gym.make("Pendulum-v0").unwrapped
observation_dims = env.observation_space.shape[0]
action_dims = env.action_space.shape[0]

env_args = EnvArgs(
    trajs=10,
    steps=200,
    epochs=120,
    batch_size=40,
    observation_dims=observation_dims,
    action_dims=action_dims
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

worker = Worker(
    env=env,
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
