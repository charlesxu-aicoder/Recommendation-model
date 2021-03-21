# -*-coding: utf-8-*-
# @Author  : Charlesxu
# @Email   : charlesxu.ai@gmail.com

from layers import *

class Attention(Layer):
    def __init__(self, att_hidden_units, activation='prelu'):

        super(Attention, self).__init__()
        self.att_dense = [Dense(unit, activation=activation) for unit in att_hidden_units]
        self.att_final_dense = Dense(1)

    def call(self, inputs):
        q, k, v, mask = inputs
        q = tf.tile(q, multiples=[1, k.shape[1]])
        q = tf.reshape(q, shape=[-1, k.shape[1], k.shape[2]])

        info = tf.concat([q, k, q - k, q * k], axis=-1)

        for dense in self.att_dense:
            info = dense(info)

        outputs = self.att_final_dense(info)
        outputs = tf.squeeze(outputs, axis=-1)

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(mask, 0), paddings, outputs)

        outputs = tf.nn.softmax(logits=outputs)
        outputs = tf.expand_dims(outputs, axis=1)

        outputs = tf.matmul(outputs, v)
        outputs = tf.squeeze(outputs, axis=1)

        return outputs


class Dice(Layer):
    def __init__(self):
        super(Dice, self).__init__()
        self.bn = BatchNormalization(center=False, scale=False)
        self.alpha = self.add_weight(shape=(), dtype=tf.float32, name='alpha')

    def call(self, x):
        x_normed = self.bn(x)
        x_p = tf.sigmoid(x_normed)

        return self.alpha * (1.0 - x_p) * x + x_p * x