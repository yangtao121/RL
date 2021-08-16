import tensorflow as tf
import tensorflow_probability as tfp


class Gaussian_policy:
    def __init__(self, Model=None):
        self.Model = Model

    # @tf.function
    def get_action(self, obs):
        mu, sigma = self.Model(obs)

        dist = tfp.distributions.Normal(mu, sigma)
        action = tf.squeeze(dist.sample(), axis=0)

        prob = tf.squeeze(dist.prob(action), axis=0)

        return action, prob

    # @tf.function
    def save_model(self, file=None):
        if file is None:
            tf.keras.models.save_model(self.Model, 'policy.h5', overwrite=True)
        else:
            tf.keras.models.save_model(self.Model, file + '/policy.h5', overwrite=True)

    def load_model(self, file):
        self.Model = tf.keras.models.load_model(filepath=file, compile=False)

    def clone_mode(self):
        model = tf.keras.models.clone_model(self.Model)
        return model

    def net_visual(self, file=None):
        self.Model.summary()
        if file is None:
            tf.keras.utils.plot_model(self.Model, 'policy.png', show_shapes=True)
        else:
            tf.keras.utils.plot_model(self.Model, file, show_shapes=True)
