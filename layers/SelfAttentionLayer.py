# -*-coding: utf-8-*-
# @Author  : Charlesxu
# @Email   : charlesxu.ai@gmail.com

from layers import *

class SelfAttention(Layer):
    def __init__(self):
        super(SelfAttention, self).__init__()

    def build(self, input_shape):
        self.dim = input_shape[0][-1]
        self.W = self.add_weight(shape=[self.dim, self.dim], name='weight',
            initializer='random_uniform')

    def call(self, inputs, **kwargs):
        q, k, v, mask = inputs
        k += self.positional_encoding(k)
        q += self.positional_encoding(q)
        q = tf.nn.relu(tf.matmul(q, self.W))
        k = tf.nn.relu(tf.matmul(k, self.W))
        mat_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(self.dim, dtype=tf.float32)
        scaled_att_logits = mat_qk / tf.sqrt(dk)
        mask = tf.tile(tf.expand_dims(mask, 1), [1, q.shape[1], 1])
        paddings = tf.ones_like(scaled_att_logits) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(mask, 0), paddings, scaled_att_logits)
        outputs = tf.nn.softmax(logits=outputs, axis=-1)
        outputs = tf.matmul(outputs, v)
        outputs = tf.reduce_mean(outputs, axis=1)
        return outputs

    @staticmethod
    def get_angles(pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def positional_encoding(self, QK_input):
        angle_rads = self.get_angles(np.arange(QK_input.shape[1])[:, np.newaxis],
                                np.arange(self.dim)[np.newaxis, :], self.dim)
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]

        return tf.cast(pos_encoding, dtype=tf.float32)

