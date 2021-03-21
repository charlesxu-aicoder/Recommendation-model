# -*-coding: utf-8-*-
# @Author  : Charlesxu
# @Email   : charlesxu.ai@gmail.com

from layers import *

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(seq_inputs, embed_dim):
    angle_rads = get_angles(np.arange(seq_inputs.shape[-1])[:, np.newaxis],
                            np.arange(embed_dim)[np.newaxis, :], embed_dim)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


def scaled_dot_product_attention(q, k, v, mask, causality=True):
    mat_qk = tf.matmul(q, k, transpose_b=True)
    dk = tf.cast(k.shape[-1], dtype=tf.float32)
    scaled_att_logits = mat_qk / tf.sqrt(dk)

    paddings = tf.ones_like(scaled_att_logits) * (-2 ** 32 + 1)
    outputs = tf.where(tf.equal(mask, 0), paddings, scaled_att_logits)

    if causality:
        diag_vals = tf.ones_like(outputs)
        masks = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()

        paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(masks, 0), paddings, outputs)

    outputs = tf.nn.softmax(logits=outputs)
    outputs = tf.matmul(outputs, v)

    return outputs


class MultiHeadAttention(Layer):
    def __init__(self, d_model, num_heads, causality=True):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.causality = causality

        self.wq = Dense(d_model, activation=None)
        self.wk = Dense(d_model, activation=None)
        self.wv = Dense(d_model, activation=None)

    def call(self, q, k, v, mask):
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = tf.reshape(tf.concat([tf.split(q, self.num_heads, axis=2)], axis=0),
                       (-1, q.shape[1], q.shape[2] // self.num_heads))
        k = tf.reshape(tf.concat([tf.split(k, self.num_heads, axis=2)], axis=0),
                       (-1, k.shape[1], k.shape[2] // self.num_heads))
        v = tf.reshape(tf.concat([tf.split(v, self.num_heads, axis=2)], axis=0),
                       (-1, v.shape[1], v.shape[2] // self.num_heads))

        scaled_attention = scaled_dot_product_attention(q, k, v, mask, self.causality)

        outputs = tf.concat(tf.split(scaled_attention, self.num_heads, axis=0), axis=2)

        return outputs

