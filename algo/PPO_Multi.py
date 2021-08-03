import tensorflow as tf
from algo.PPO import PPO
import multiprocessing as mp
import gym
from common.worker import Worker
import time
from common.functions import mkdir
from common.critic import Critic
from common.policy import Gaussian_policy


class PPO_Multi(PPO):
    def __init__(self,
                 policy,
                 critic,
                 env_args,
                 hyper_parameter,
                 net_visualize=False
                 ):
        self.policy = policy
        self.critic = critic
        self.clip_ratio = hyper_parameter.clip_ratio
        self.update_steps = hyper_parameter.update_steps

        self.critic_optimizer = tf.optimizers.Adam(learning_rate=hyper_parameter.critic_learning_rate)
        self.policy_optimizer = tf.optimizers.Adam(learning_rate=hyper_parameter.policy_learning_rate)

        self.batch_size = env_args.batch_size
        self.span = env_args.span
        self.epochs = env_args.epochs

        if net_visualize:
            self.policy.net_visual()
            self.critic.net_visual()

        self.UPDATE_EVENT, self.ROLLING_EVENT = mp.Event(), mp.Event()
        self.UPDATE_EVENT.clear()
        self.UPDATE_EVENT.clear()

        self.multi_worker_num = env_args.multi_worker_num

        self.batches = mp.Queue(self.multi_worker_num)

        self.env_args = env_args
        self.hyper_parameter = hyper_parameter

    def rolling(self, env_name):
        env = gym.make(env_name).unwrapped
        worker = Worker(
            env=env,
            env_args=self.env_args,
            hyper_parameter=self.hyper_parameter
        )
        policy = Gaussian_policy()
        critic = Critic()

        for _ in range(self.epochs):
            policy.load_model('data/policy.h5')
            critic.load_model('data/critic.h5')
            self.ROLLING_EVENT.wait()
            worker.update(policy, critic)
            batch = worker.runner()
            self.batches.put(batch)
            if self.batches.qsize() < self.multi_worker_num - 1:
                self.UPDATE_EVENT.wait()
            # if self.roll_flag < self.multi_worker_num:

            else:
                self.roll_flag = 0
                self.UPDATE_EVENT.set()
                self.ROLLING_EVENT.clear()

    def train(self, env_name, path=None):
        if path is None:
            path = 'data'
            mkdir(path)
        else:
            mkdir(path)

        self.policy.save_model(path)
        self.critic.save_model(path)

        threads = [mp.Process(target=self.rolling, args=[env_name]) for _ in range(self.multi_worker_num)]
        # threads.append(mp.Process(target=self.run_optimize))

        for thread in threads:
            thread.start()
        self.UPDATE_EVENT.clear()
        self.ROLLING_EVENT.set()
        time.sleep(0.1)

        for j in range(self.epochs):
            print("---------------------obtain samples:{}---------------------".format(j))
            time_start = time.time()
            self.UPDATE_EVENT.wait()
            time_end = time.time()
            print('consuming time:{}'.format(time_end - time_start))
            batches = []
            for i in range(self.multi_worker_num):
                batches = batches + self.batches.get()
            self.optimize(batches)
            self.policy.save_model(path)
            self.critic.save_model(path)
            self.ROLLING_EVENT.set()
            self.UPDATE_EVENT.clear()
            print("----------------------------------------------------------")
