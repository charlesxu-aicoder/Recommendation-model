# -*-coding: utf-8-*-
# @Author  : Charlesxu
# @Email   : charlesxu.ai@gmail.com

from models.matching import *
from layers.DnnLayer import *

class PNN(keras.Model):
    def __init__(self, feature_columns, hidden_units, mode='in', dnn_dropout=0.,
                 activation='relu', embed_reg=1e-4, w_z_reg=1e-4, w_p_reg=1e-4, l_b_reg=1e-4):
        super(PNN, self).__init__()
        self.mode = mode
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.field_num = len(self.sparse_feature_columns)
        self.embed_dim = self.sparse_feature_columns[0]['embed_dim']
        self.embed_layers = {
            'embed_' + str(i): Embedding(input_dim=feat['feat_num'],
                                         input_length=1,
                                         output_dim=feat['embed_dim'],
                                         embeddings_initializer='random_uniform',
                                         embeddings_regularizer=l2(embed_reg))
            for i, feat in enumerate(self.sparse_feature_columns)
        }
        self.w_z = self.add_weight(name='w_z',
                                   shape=(self.field_num, self.embed_dim, hidden_units[0]),
                                   initializer='random_uniform',
                                   regularizer=l2(w_z_reg),
                                   trainable=True
                                   )
        if mode == 'in':
            self.w_p = self.add_weight(name='w_p',
                                       shape=(self.field_num * (self.field_num - 1) // 2, self.embed_dim,
                                              hidden_units[0]),
                                       initializer='random_uniform',
                                       reguarizer=l2(w_p_reg),
                                       trainable=True)
        else:
            self.w_p = self.add_weight(name='w_p',
                                       shape=(self.field_num * (self.field_num - 1) // 2, self.embed_dim,
                                              self.embed_dim, hidden_units[0]),
                                       initializer='random_uniform',
                                       regularizer=l2(w_p_reg),
                                       trainable=True)
        self.l_b = self.add_weight(name='l_b', shape=(hidden_units[0], ),
                                   initializer='random_uniform',
                                   regularizer=l2(l_b_reg),
                                   trainable=True)
        self.dnn_network = DNN(hidden_units[1:], activation, dnn_dropout)
        self.dense_final = Dense(1)

    def call(self, inputs):
        dense_inputs, sparse_inputs = inputs
        embed = [self.embed_layers['embed_{}'.format(i)](sparse_inputs[:, i])
                 for i in range(sparse_inputs.shape[1])]
        embed = tf.transpose(tf.convert_to_tensor(embed), [1, 0, 2])  # (None, field_num, embed_dim)
        row = []
        col = []
        for i in range(len(self.sparse_feature_columns) - 1):
            for j in range(i + 1, len(self.sparse_feature_columns)):
                row.append(i)
                col.append(j)
        p = tf.gather(embed, row, axis=1)
        q = tf.gather(embed, col, axis=1)
        if self.mode == 'in':
            l_p = tf.tensordot(p*q, self.w_p, axes=2)
        else:
            u = tf.expand_dims(q, 2)
            v = tf.expand_dims(p, 2)
            l_p = tf.tensordot(tf.matmul(tf.transpose(u, [0, 1, 3, 2]), v), self.w_p, axes=3)

        l_z = tf.tensordot(embed, self.w_z, axes=2)
        l_1 = tf.nn.relu(tf.concat([l_z + l_p + self.l_b, dense_inputs], axis=-1))
        dnn_x = self.dnn_network(l_1)
        outputs = tf.nn.sigmoid(self.dense_final(dnn_x))
        return outputs

    def summary(self):
        dense_inputs = Input(shape=(len(self.dense_feature_columns),), dtype=tf.float32)
        sparse_inputs = Input(shape=(len(self.sparse_feature_columns),), dtype=tf.int32)
        keras.Model(inputs=[dense_inputs, sparse_inputs],
                    outputs=self.call([dense_inputs, sparse_inputs])).summary()