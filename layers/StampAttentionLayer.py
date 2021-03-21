# -*-coding: utf-8-*-
# @Author  : Charlesxu
# @Email   : charlesxu.ai@gmail.com

from layers import *

class StampAttention(Layer):

    def __init__(self, d, reg=1e-4):
        self.d = d
        self.reg = reg
        super(StampAttention, self).__init__()

    def build(self, input_shape):
        self.W0 = self.add_weight(name='W0',
                                  shape=(self.d, 1),
                                  initializer=tf.random_normal_initializer,
                                  regularizer=l2(self.reg),
                                  trainable=True)
        self.W1 = self.add_weight(name='W1',
                                  shape=(self.d, self.d),
                                  initializer=tf.random_normal_initializer,
                                  regularizer=l2(self.reg),
                                  trainable=True)
        self.W2 = self.add_weight(name='W2',
                                  shape=(self.d, self.d),
                                  initializer=tf.random_normal_initializer,
                                  regularizer=l2(self.reg),
                                  trainable=True)
        self.W3 = self.add_weight(name='W3',
                                  shape=(self.d, self.d),
                                  initializer=tf.random_normal_initializer,
                                  regularizer=l2(self.reg),
                                  trainable=True)
        self.b = self.add_weight(name='b',
                                  shape=(self.d,),
                                  initializer=tf.random_normal_initializer,
                                  regularizer=l2(self.reg),
                                  trainable=True)

    def call(self, inputs):
        seq_embed, m_s, x_t = inputs
        alpha = tf.matmul(tf.nn.sigmoid(
            tf.tensordot(seq_embed, self.W1, axes=[2, 0]) + tf.expand_dims(tf.matmul(x_t, self.W2), axis=1) +
            tf.expand_dims(tf.matmul(m_s, self.W3), axis=1) + self.b), self.W0)
        m_a = tf.reduce_sum(tf.multiply(alpha, seq_embed), axis=1)  # (None, d)
        return m_a
