import numpy as np
import os


def cal_simple_adv(policy, state, discount_reward):
    adv = discount_reward - policy(state)
    return adv.numpy()


def cal_discount_r(gamma, buffer_reward, value_):
    discount_reward = []
    for reward in buffer_reward[::-1]:
        value_ = reward + gamma * value_
        discount_reward.append(value_)
    discount_reward.reverse()
    return discount_reward


def cal_adv(gamma, lambada, values, reward, last_v):
    """

    :param lambada:
    :param gamma:
    :param values: 一个batch中的value
    :param reward:
    :param last_v: t+1时候的value
    :return:
    """
    values = np.append(values, last_v)
    deltas = reward + gamma * values[1:] - values[:-1]
    advantage = 0

    advantages = []
    for delta in deltas[::-1]:
        advantage = delta + gamma * lambada * advantage
        advantages.append(advantage)

    return advantages


def gae_target(gamma, lmbda, rewards, v_values, next_v_value, done):
    n_step_targets = np.zeros_like(rewards)
    gae = np.zeros_like(rewards)
    gae_cumulative = 0
    forward_val = 0

    if not done:
        forward_val = next_v_value

    for k in reversed(range(0, len(rewards))):
        delta = rewards[k] + gamma * forward_val - v_values[k]
        gae_cumulative = gamma * lmbda * gae_cumulative + delta
        gae[k] = gae_cumulative
        forward_val = v_values[k]
        n_step_targets[k] = gae[k] + v_values[k]
    return gae, n_step_targets


def standardize(data):
    mean, std = (
        np.mean(data),
        np.std(data)
    )
    data = (data - mean) / std

    return data


def mkdir(path):
    current = os.getcwd()
    path = current+'/'+path
    flag = os.path.exists(path)
    if flag is False:
        os.mkdir(path)
