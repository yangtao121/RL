class EnvArgs:
    def __init__(self, trajs, steps, epochs, batch_size, mini_batch_size_num, observation_dims, action_dims,
                 multi_worker_num=None):
        self.trajs = trajs
        self.steps = steps
        self.batch_size = batch_size

        self.observation_dims = observation_dims
        self.action_dims = action_dims
        self.epochs = epochs
        self.span = int((trajs * steps) / mini_batch_size_num)

        if multi_worker_num is not None:
            self.span = int((multi_worker_num * trajs * steps) / batch_size)
            self.multi_worker_num = multi_worker_num


class HyperParameter:
    def __init__(self,
                 clip_ratio=0.2,
                 policy_learning_rate=3e-4,
                 critic_learning_rate=1e-3,
                 update_steps=10,
                 gamma=0.99,
                 lambada=0.95
                 ):
        self.clip_ratio = clip_ratio
        self.policy_learning_rate = policy_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.update_steps = update_steps
        self.gamma = gamma
        self.lambada = lambada
