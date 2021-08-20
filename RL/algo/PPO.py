import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import time
from RL.common.functions import mkdir, standardize, gae_target
from matplotlib import pyplot as plt
from RL.common.utilit import RewardScale


class PPO:
    def __init__(self,
                 policy,
                 critic,
                 worker,
                 env_args,
                 hyper_parameter,
                 net_visualize=False,
                 ):
        tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()
        self.policy = policy
        self.critic = critic
        self.worker = worker

        self.clip_ratio = hyper_parameter.clip_ratio
        self.update_steps = hyper_parameter.update_steps
        self.tolerance = hyper_parameter.tolerance
        self.center = hyper_parameter.center
        self.reward_scale = hyper_parameter.reward_scale
        self.scale = hyper_parameter.scale
        self.clip_value = hyper_parameter.clip_value
        self.gamma = hyper_parameter.gamma
        self.lambada = hyper_parameter.lambada
        self.center_adv = hyper_parameter.center_adv

        self.critic_optimizer = tf.optimizers.Adam(learning_rate=hyper_parameter.critic_learning_rate)
        self.policy_optimizer = tf.optimizers.Adam(learning_rate=hyper_parameter.policy_learning_rate)

        self.batch_size = env_args.batch_size
        self.span = env_args.span
        self.epochs = env_args.epochs
        self.steps = env_args.steps
        self.total_steps = env_args.total_steps

        self.mini_batch_size_num = env_args.mini_batch_size_num
        self.mini_batch_size = env_args.mini_batch_size

        if self.reward_scale:
            self.RewardScale = RewardScale(self.total_steps, self.center, self.scale)

        if net_visualize:
            self.policy.net_visual()
            self.critic.net_visual()

    @tf.function
    def critic_train(self, observation, target):
        observation = tf.convert_to_tensor(observation, dtype=tf.float32)
        target = tf.convert_to_tensor(target, dtype=tf.float32)
        if self.clip_value:
            with tf.GradientTape() as tape:
                v = self.critic.Model(observation)
                surrogate1 = tf.square(v[1:] - target[1:])
                surrogate2 = tf.square(
                    tf.clip_by_value(v[1:], v[:-1] - self.clip_ratio, v[:-1] + self.clip_ratio) - target[1:])
                critic_loss = tf.reduce_mean(tf.minimum(surrogate1, surrogate2))
        else:
            with tf.GradientTape() as tape:
                v = self.critic.Model(observation)
                critic_loss = tf.reduce_mean(tf.square(target - v))

        grad = tape.gradient(critic_loss, self.critic.Model.trainable_weights)
        self.critic_optimizer.apply_gradients(zip(grad, self.critic.Model.trainable_weights))

    #
    # @tf.function
    # def critic_train(self, state, discount_reward):
    #     with tf.GradientTape() as tape:
    #         v = self.critic.Model(state)
    #         critic_loss = tf.reduce_mean(tf.square(discount_reward - v))
    #
    #     grad = tape.gradient(critic_loss, self.critic.Model.trainable_weights)
    #     self.critic_optimizer.apply_gradients(zip(grad, self.critic.Model.trainable_weights))

    @tf.function
    def policy_train(self, state, action, advantage, old_prob):
        """

        :param state:
        :param action:
        :param advantage:
        :param old_prob: old actor net output
        :return:
        """
        with tf.GradientTape() as tape:
            # 计算新的网络分布
            mu, sigma = self.policy.Model(state)
            pi = tfp.distributions.Normal(mu, sigma)

            ratio = pi.prob(action) / (old_prob + 1e-8)
            del pi
            # ratio = old_prob

            actor_loss = -tf.reduce_mean(
                tf.minimum(
                    ratio * advantage,
                    tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantage
                )
            )
        actor_grad = tape.gradient(actor_loss, self.policy.Model.trainable_weights)
        self.policy_optimizer.apply_gradients(zip(actor_grad, self.policy.Model.trainable_weights))
        return actor_loss

    @tf.function
    def get_loss(self, state, action, advantage, old_prob, discount_reward):

        mu, sigma = self.policy.Model(state)
        pi = tfp.distributions.Normal(mu, sigma)

        ratio = pi.prob(action) / (old_prob + 1e-8)
        del pi

        v = self.critic.Model(state)
        surrogate1 = tf.square(v[1:] - discount_reward[1:])
        surrogate2 = tf.square(
            tf.clip_by_value(v[1:], v[:-1] - self.clip_ratio, v[:-1] + self.clip_ratio) - discount_reward[1:])
        critic_loss = tf.reduce_mean(tf.minimum(surrogate1, surrogate2))
        # critic_loss = tf.reduce_mean(tf.square(discount_reward - v))

        actor_loss = -tf.reduce_mean(
            tf.minimum(
                ratio * advantage,
                tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantage
            )
        )

        return actor_loss, critic_loss

    @tf.function
    def get_v(self, obs):
        return tf.squeeze(self.critic.Model(obs))

    def optimize(self, batches):
        sum_rewards = []
        for batch in batches:
            sum_reward = np.sum(batch.reward_buffer)
            sum_rewards.append(sum_reward)
        sum_rewards = np.hstack(sum_rewards)
        info = {
            'max': np.max(sum_rewards),
            'min': np.min(sum_rewards),
            'avg': np.mean(sum_rewards)
        }
        print("Max episode reward:{}".format(info['max']))
        print("Min episode reward:{}".format(info["min"]))
        print("Average episode reward:{}".format(info["avg"]))

        observation_buffer = np.concatenate([batch.observation_buffer for batch in batches])
        next_observation_buffer = np.concatenate([batch.next_observation_buffer for batch in batches])
        reward_buffer = np.concatenate([batch.reward_buffer for batch in batches])
        action_buffer = np.concatenate([batch.action_buffer for batch in batches])
        old_probs = np.concatenate([batch.prob_buffer for batch in batches])
        # tf_next_observation_buffer = tf.cast(next_observation_buffer, dtype=tf.float32)

        values = self.get_v(observation_buffer).numpy()
        values_ = self.get_v(next_observation_buffer).numpy()

        sum_batch_rewards = []

        for i in range(self.span):
            path = slice(i * self.batch_size, (i + 1) * self.batch_size)
            sum_rewards = np.sum(reward_buffer[path])
            sum_batch_rewards.append(sum_rewards)

        print("Max mini_batch reward:{}".format(np.max(sum_batch_rewards)))
        print("Min mini_batch reward:{}".format(np.min(sum_batch_rewards)))
        print("Average mini_batch reward:{}".format(np.mean(sum_batch_rewards)))

        if self.reward_scale:
            reward_buffer = self.RewardScale(reward_buffer)

        gaes = []
        targets = []
        for i in range(self.span):
            path = slice(i * self.batch_size, (i + 1) * self.batch_size)
            gae, target = gae_target(self.gamma, self.lambada, reward_buffer[path], values[path],
                                     values_[i * self.batch_size - 1], done=False)

            gaes.append(gae)
            targets.append(target)
        gaes = np.concatenate([gae for gae in gaes])
        targets = np.concatenate([target for target in targets])

        if self.center_adv:
            gaes = standardize(gaes)

        action_buffer = tf.cast(action_buffer, dtype=tf.float32)
        gaes = tf.cast(gaes, dtype=tf.float32)
        old_probs = tf.cast(old_probs, dtype=tf.float32)
        targets = tf.cast(targets, dtype=tf.float32)
        tf_observation_buffer = tf.cast(observation_buffer, dtype=tf.float32)

        # actor_loss_before, critic_loss_before = self.get_loss(observation_buffer, action_buffer, gaes, old_probs,
        #                                                       targets)
        # print('loss before, actor:{},critic:{}'.format(actor_loss_before, critic_loss_before))
        last_loss = 0
        now_loss = 1
        update_policy_flag = True
        for _ in tf.range(0, self.update_steps):
            for i in tf.range(0, self.mini_batch_size_num):
                path = slice(i * self.mini_batch_size, (i + 1) * self.mini_batch_size)
                state = tf_observation_buffer[path]
                action = action_buffer[path]
                gae = gaes[path]
                old_prob = old_probs[path]
                target = targets[path]
                if update_policy_flag:
                    now_loss = self.policy_train(state, action, gae, old_prob)
                self.critic_train(state, target)
                if abs(now_loss - last_loss) < self.tolerance:
                    update_policy_flag = False
                    # print(update_policy_flag)
                last_loss = now_loss

        del batches[:]
        del observation_buffer
        del action_buffer
        del gaes
        del old_probs
        del targets
        return info

    # def optimize(self, batches):
    #     sum_rewards = []
    #     for batch in batches:
    #         sum_reward = np.sum(batch.reward_buffer)
    #         sum_rewards.append(sum_reward)
    #     sum_rewards = np.hstack(sum_rewards)
    #     info = {
    #         'max': np.max(sum_rewards),
    #         'min': np.min(sum_rewards),
    #         'avg': np.mean(sum_rewards)
    #     }
    #     print("Max episode reward:{}".format(info['max']))
    #     print("Min episode reward:{}".format(info["min"]))
    #     print("Average episode reward:{}".format(info["avg"]))
    #
    #     observation_buffer = np.concatenate([batch.observation_buffer for batch in batches])
    #     reward_buffer = np.concatenate([batch.reward_buffer for batch in batches])
    #     action_buffer = np.concatenate([batch.action_buffer for batch in batches])
    #     gaes = np.concatenate([batch.advantage_buffer for batch in batches])
    #     old_probs = np.concatenate([batch.prob_buffer for batch in batches])
    #     targets = np.concatenate([batch.target_buffer for batch in batches])
    #
    #     sum_batch_rewards = []
    #     for i in range(self.span):
    #         path = slice(i * self.batch_size, (i + 1) * self.batch_size)
    #         sum_rewards = np.sum(reward_buffer[path])
    #         sum_batch_rewards.append(sum_rewards)
    #
    #     print("Max mini_batch reward:{}".format(np.max(sum_batch_rewards)))
    #     print("Min mini_batch reward:{}".format(np.min(sum_batch_rewards)))
    #     print("Average mini_batch reward:{}".format(np.mean(sum_batch_rewards)))
    #
    #     if self.center_adv:
    #         gaes = standardize(gaes)
    #         # print('using ')
    #
    #     observation_buffer = tf.cast(observation_buffer, dtype=tf.float32)
    #     action_buffer = tf.cast(action_buffer, dtype=tf.float32)
    #     gaes = tf.cast(gaes, dtype=tf.float32)
    #     old_probs = tf.cast(old_probs, dtype=tf.float32)
    #     targets = tf.cast(targets, dtype=tf.float32)
    #
    #     # actor_loss_before, critic_loss_before = self.get_loss(observation_buffer, action_buffer, gaes, old_probs,
    #     #                                                       targets)
    #     # print('loss before, actor:{},critic:{}'.format(actor_loss_before, critic_loss_before))
    #     last_loss = 0
    #     now_loss = 1
    #     update_policy_flag = True
    #     for _ in tf.range(0, self.update_steps):
    #         for i in tf.range(0, self.mini_batch_size_num):
    #             path = slice(i * self.mini_batch_size, (i + 1) * self.mini_batch_size)
    #             state = observation_buffer[path]
    #             action = action_buffer[path]
    #             gae = gaes[path]
    #             old_prob = old_probs[path]
    #             target = targets[path]
    #             if update_policy_flag:
    #                 now_loss = self.policy_train(state, action, gae, old_prob)
    #             self.critic_train(state, target)
    #             if abs(now_loss - last_loss) < self.tolerance:
    #                 update_policy_flag = False
    #                 # print(update_policy_flag)
    #             last_loss = now_loss
    #
    #     del batches[:]
    #     del observation_buffer
    #     del action_buffer
    #     del gaes
    #     del old_probs
    #     del targets
    #     return info

    def train(self, path=None, title=None):
        if path is None:
            path = 'data'
            mkdir(path)
        else:
            mkdir(path)

        average_reward = []

        for i in range(self.epochs):
            print("---------------------obtain samples:{}---------------------".format(i))
            self.worker.update(self.policy, self.critic)

            time_start = time.time()
            batches = self.worker.runner()
            time_end = time.time()
            print('consuming time:{}'.format(time_end - time_start))

            info = self.optimize(batches)
            average_reward.append(info['avg'])
            self.policy.save_model(path)
            self.critic.save_model(path)
            print("----------------------------------------------------------")

        plt.plot(average_reward)
        plt.xlabel('episode')
        plt.ylabel('avg_reward')
        if title:
            plt.title(title)
        plt.show()
