from common.collector import Collector
from common.functions import gae_target

import queue

from multiprocessing import Process
from multiprocessing import Queue


class Worker:

    def __init__(self, env, env_args, hyper_parameter):
        self.env = env
        self.trajs = env_args.trajs
        self.batch_size = env_args.batch_size
        self.steps = env_args.steps

        self.policy = None
        self.critic = None

        self.obs_dims = env_args.observation_dims
        self.act_dims = env_args.action_dims

        self.gamma = hyper_parameter.gamma
        self.lambada = hyper_parameter.lambada

    def update(self, policy, critic):
        """
        使用新的policy和critic
        :param policy:
        :param critic:
        :return:
        """

        self.policy = policy
        self.critic = critic

    def runner(self):
        # print('start')
        batches = []
        for i in range(self.trajs):
            collector = Collector(observation_dims=self.obs_dims, action_dims=self.act_dims,
                                  episode_length=self.steps)
            state = self.env.reset()
            # print(i)

            for t in range(self.steps):
                state = state.reshape(1, -1)
                action, prob = self.policy.get_action(state)

                action_ = action * 2

                state_, reward, done, _ = self.env.step(action_)
                collector.store(state, action, reward, prob)
                state = state_

                if (t + 1) % self.batch_size == 0 or t == self.steps - 1:
                    observations, reward = collector.get_current_data()
                    value_ = self.critic.get_value(state_.reshape(1, -1))
                    values = self.critic.get_value(observations)

                    gae, target = gae_target(self.gamma, self.lambada, reward, values, value_, done)

                    collector.get_gae_target(gae, target)

            batches.append(collector)

        return batches


class Worker2:
    def __init__(self, env_args, hyper_parameter):
        self.trajs = env_args.trajs
        self.batch_size = env_args.batch_size
        self.steps = env_args.steps

        self.policy = None
        self.critic = None

        self.obs_dims = env_args.observation_dims
        self.act_dims = env_args.action_dims

        self.gamma = hyper_parameter.gamma
        self.lambada = hyper_parameter.lambada

    def update(self, policy, critic):
        """
        使用新的policy和critic
        :param policy:
        :param critic:
        :return:
        """

        self.policy = policy
        self.critic = critic

    def runner(self, env):
        # print('start')
        batches = []
        for i in range(self.trajs):
            collector = Collector(observation_dims=self.obs_dims, action_dims=self.act_dims,
                                  episode_length=self.steps)
            state = env.reset()
            # print(i)

            for t in range(self.steps):
                state = state.reshape(1, -1)
                action, prob = self.policy.get_action(state)

                action_ = action * 2

                state_, reward, done, _ = env.step(action_)
                collector.store(state, action, reward, prob)
                state = state_

                if (t + 1) % self.batch_size == 0 or t == self.steps - 1:
                    observations, reward = collector.get_current_data()
                    value_ = self.critic.get_value(state_.reshape(1, -1))
                    values = self.critic.get_value(observations)

                    gae, target = gae_target(self.gamma, self.lambada, reward, values, value_, done)

                    collector.get_gae_target(gae, target)

            batches.append(collector)

        return batches


class MultiWorker:
    def __init__(self, envs, env_args, hyper_parameter):
        self.workers = []
        for env in envs:
            worker = Worker(env, env_args, hyper_parameter)
            self.workers.append(worker)

        self.multi_worker_num = env_args.multi_worker_num

        self.q = Queue(env_args.multi_worker_num)

    def update(self, policy, critic):
        for worker in self.workers:
            worker.update(policy, critic)

    def runner(self):
        threads = [Process(target=self.runner_single, args=[i]) for i in range(self.multi_worker_num)]

        for thread in threads:
            thread.start()

        # for thread in threads:
        #     thread.join()

        batches = []

        # for worker in self.workers:
        #     batches = batches + worker.batches

        for _ in range(self.multi_worker_num):
            batches = batches + self.q.get()

        return batches

    def runner_single(self, index):
        batch = self.workers[index].runner()
        self.q.put(batch)
        # print(batch)
