import tensorflow as tf


class Critic:
    def __init__(self, Model=None):
        self.Model = Model

    def get_value(self, obs):
        v = self.Model(obs)
        v = tf.squeeze(v)
        return v

    def save_model(self, file=None):
        if file is None:
            tf.keras.models.save_model(self.Model, 'critic.h5', overwrite=True)
        else:
            tf.keras.models.save_model(self.Model, filepath=file + '/critic.h5', overwrite=True)

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
