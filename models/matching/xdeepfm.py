# -*-coding: utf-8-*-
# @Author  : Charlesxu
# @Email   : charlesxu.ai@gmail.com

from models.matching import *
from layers.LinearLayer import *
from layers.DnnLayer import *

class CIN(layers.Layer):
    def __init__(self, cin_size, l2_reg=1e-4):
        super(CIN, self).__init__()
        self.cin_size = cin_size
        self.l2_reg = l2_reg

    def build(self, input_shape):
        self.embedding_nums = input_shape[1]
        self.field_nums = [self.embedding_nums] + self.cin_size
        self.cin_W = {
            'CIN_W_' + str(i): self.add_weight(
                name='CIN_W_' + str(i),
                shape=(1, self.field_nums[0] * self.field_nums[i], self.field_nums[i + 1]),
                initializer='random_uniform',
                regularizer=l2(self.l2_reg),
                trainable=True)
            for i in range(len(self.field_nums) - 1)
        }

    def call(self, inputs, **kwargs):
        dim = inputs.shape[-1]
        hidden_layers_results = [inputs]
        split_X_0 = tf.split(hidden_layers_results[0], dim, 2)
        for idx, size in enumerate(self.cin_size):
            split_X_K = tf.split(hidden_layers_results[-1], dim, 2)

            result_1 = tf.matmul(split_X_0, split_X_K, transpose_b=True)

            result_2 = tf.reshape(result_1, shape=[dim, -1, self.embedding_nums * self.field_nums[idx]])

            result_3 = tf.transpose(result_2, perm=[1, 0, 2])

            result_4 = tf.nn.conv1d(input=result_3, filters=self.cin_W['CIN_W_' + str(idx)], stride=1,
                                    padding='VALID')

            result_5 = tf.transpose(result_4, perm=[0, 2, 1])

            hidden_layers_results.append(result_5)

        final_results = hidden_layers_results[1:]
        result = tf.concat(final_results, axis=1)
        result = tf.reduce_sum(result,  axis=-1)

        return result


class xDeepFM(keras.Model):
    def __init__(self, feature_columns, hidden_units, cin_size, dnn_dropout=0, dnn_activation='relu',
                 embed_reg=1e-5, cin_reg=1e-5):
        super(xDeepFM, self).__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.embed_dim = self.sparse_feature_columns[0]['embed_dim']
        self.embed_layers = {
            'embed_' + str(i): Embedding(input_dim=feat['feat_num'],
                                         input_length=1,
                                         output_dim=feat['embed_dim'],
                                         embeddings_initializer='random_uniform',
                                         embeddings_regularizer=l2(embed_reg))
            for i, feat in enumerate(self.sparse_feature_columns)
        }
        self.linear = Linear()
        self.cin = CIN(cin_size=cin_size, l2_reg=cin_reg)
        self.dnn = DNN(hidden_units=hidden_units, dnn_dropout=dnn_dropout, dnn_activation=dnn_activation)
        self.cin_dense = Dense(1)
        self.dnn_dense = Dense(1)
        self.bias = self.add_weight(name='bias', shape=(1, ), initializer=tf.zeros_initializer())

    def call(self, inputs, **kwargs):
        dense_inputs, sparse_inputs = inputs

        embed = [self.embed_layers['embed_{}'.format(i)](sparse_inputs[:, i]) for i in range(sparse_inputs.shape[1])]
        embed_matrix = tf.transpose(tf.convert_to_tensor(embed), [1, 0, 2])
        cin_out = self.cin(embed_matrix)
        cin_out = self.cin_dense(cin_out)
        embed_vector = tf.reshape(embed_matrix, shape=(-1, embed_matrix.shape[1] * embed_matrix.shape[2]))
        dnn_out = self.dnn(embed_vector)
        dnn_out = self.dnn_dense(dnn_out)

        output = tf.nn.sigmoid(cin_out + dnn_out + self.bias)
        return output

    def summary(self):
        dense_inputs = Input(shape=(len(self.dense_feature_columns),), dtype=tf.float32)
        sparse_inputs = Input(shape=(len(self.sparse_feature_columns),), dtype=tf.int32)
        keras.Model(inputs=[dense_inputs, sparse_inputs], outputs=self.call([dense_inputs, sparse_inputs])).summary()