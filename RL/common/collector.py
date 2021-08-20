import numpy as np


class Collector:
    def __init__(self, observation_dims, action_dims, episode_length):
        self.observation_buffer = np.zeros((episode_length, observation_dims), dtype=np.float32)
        self.next_observation_buffer = np.zeros((episode_length, observation_dims), dtype=np.float32)

        self.action_buffer = np.zeros((episode_length, action_dims), dtype=np.float32)
        self.reward_buffer = np.zeros((episode_length, 1), dtype=np.float32)

        self.advantage_buffer = np.zeros((episode_length, 1), dtype=np.float32)
        self.target_buffer = np.zeros((episode_length, 1), dtype=np.float32)

        self.prob_buffer = np.zeros((episode_length, action_dims), dtype=np.float32)

        self.pointer = 0
        self.last_pointer = 0

    def store(self, observation, next_observation, action, reward, prob):
        self.observation_buffer[self.pointer] = observation
        self.next_observation_buffer[self.pointer] = next_observation
        self.action_buffer[self.pointer] = action
        self.reward_buffer[self.pointer] = reward
        self.prob_buffer[self.pointer] = prob

        self.pointer += 1

    def get_gae_target(self, gae, target):
        path = slice(self.last_pointer, self.pointer)
        self.advantage_buffer[path] = gae
        self.target_buffer[path] = target
        self.last_pointer = self.pointer

    def get_current_data(self):
        path = slice(self.last_pointer, self.pointer)
        observation = self.observation_buffer[path]
        reward = self.reward_buffer[path]
        return observation, reward
