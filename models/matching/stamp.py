# -*-coding: utf-8-*-
# @Author  : Charlesxu
# @Email   : charlesxu.ai@gmail.com

from models.matching import *
from layers.StampAttentionLayer import *

class STAMP(tf.keras.Model):
    def __init__(self, feature_columns, behavior_feature_list, item_pooling, maxlen=40, activation='tanh',
                 embed_reg=1e-4):
        super(STAMP, self).__init__()
        self.maxlen = maxlen

        self.item_pooling = item_pooling
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns

        self.other_sparse_len = len(self.sparse_feature_columns) - len(behavior_feature_list)
        self.dense_len = len(self.dense_feature_columns)
        self.seq_len = len(behavior_feature_list)

        self.embed_dim = self.sparse_feature_columns[0]['embed_dim']

        self.embed_sparse_layers = [Embedding(input_dim=feat['feat_num'],
                                              input_length=1,
                                              output_dim=feat['embed_dim'],
                                              embeddings_initializer='random_uniform',
                                              embeddings_regularizer=l2(embed_reg))
                                    for feat in self.sparse_feature_columns
                                    if feat['feat'] not in behavior_feature_list]
        self.embed_seq_layers = [Embedding(input_dim=feat['feat_num'],
                                           input_length=1,
                                           output_dim=feat['embed_dim'],
                                           embeddings_initializer='random_uniform',
                                           embeddings_regularizer=l2(embed_reg))
                                 for feat in self.sparse_feature_columns
                                 if feat['feat'] in behavior_feature_list]

        self.attention_layer = StampAttention(d=self.embed_dim)

        self.ffn1 = Dense(self.embed_dim, activation=activation)
        self.ffn2 = Dense(self.embed_dim, activation=activation)

    def call(self, inputs):
        dense_inputs, sparse_inputs, seq_inputs = inputs

        x = dense_inputs
        for i in range(self.other_sparse_len):
            x = tf.concat([x, self.embed_sparse_layers[i](sparse_inputs[:, i])], axis=-1)

        seq_embed, m_t, item_pooling_embed = None, None, None
        for i in range(self.seq_len):
            seq_embed = self.embed_seq_layers[i](seq_inputs[:, i]) if seq_embed is None \
                else seq_embed + self.embed_seq_layers[i](seq_inputs[:, i])
            m_t = self.embed_seq_layers[i](seq_inputs[:, i, -1]) if m_t is None \
                else m_t + self.embed_seq_layers[i](seq_inputs[-1, i, -1])  # (None, d)
            item_pooling_embed = self.embed_seq_layers[i](self.item_pooling[:, i]) \
                if item_pooling_embed is None \
                else item_pooling_embed + self.embed_seq_layers[i](self.item_pooling[:, i])  # (m, d)

        m_s = tf.reduce_mean(seq_embed, axis=1)  # (None, d)

        m_a = self.attention_layer([seq_embed, m_s, m_t])  # (None, d)

        if self.other_sparse_len != 0 or self.dense_len != 0:
            m_a = tf.concat([m_a, x], axis=-1)
            m_t = tf.concat([m_t, x], axis=-1)

        h_s = self.ffn1(m_a)
        h_t = self.ffn2(m_t)

        z = tf.matmul(tf.multiply(tf.expand_dims(h_t, axis=1), item_pooling_embed), tf.expand_dims(h_s, axis=-1))
        z = tf.squeeze(z, axis=-1)

        outputs = tf.nn.softmax(z)
        return outputs

    def summary(self):
        dense_inputs = Input(shape=(self.dense_len,), dtype=tf.float32)
        sparse_inputs = Input(shape=(self.other_sparse_len,), dtype=tf.int32)
        seq_inputs = Input(shape=(self.seq_len, self.maxlen), dtype=tf.int32)
        tf.keras.Model(inputs=[dense_inputs, sparse_inputs, seq_inputs],
                       outputs=self.call([dense_inputs, sparse_inputs, seq_inputs])).summary()
