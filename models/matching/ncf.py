# -*-coding: utf-8-*-
# @Author  : Charlesxu
# @Email   : charlesxu.ai@gmail.com

from models.matching import *
from layers.DnnLayer import *

class NCF(Model):
    def __init__(self, feature_columns, hidden_units=None, dropout=0.2, activation='relu', embed_reg=1e-6, **kwargs):
        super(NCF, self).__init__(**kwargs)
        if hidden_units is None:
            hidden_units = [64, 32, 16, 8]
        self.user_fea_col, self.item_fea_col = feature_columns
        self.mf_user_embedding = Embedding(input_dim=self.user_fea_col['feat_num'],
                                           input_length=1,
                                           output_dim=self.user_fea_col['embed_dim'],
                                           embeddings_initializer='random_normal',
                                           embeddings_regularizer=l2(embed_reg))
        self.mf_item_embedding = Embedding(input_dim=self.item_fea_col['feat_num'],
                                           input_length=1,
                                           output_dim=self.item_fea_col['embed_dim'],
                                           embeddings_initializer='random_normal',
                                           embeddings_regularizer=l2(embed_reg))
        self.mlp_user_embedding = Embedding(input_dim=self.user_fea_col['feat_num'],
                                            input_length=1,
                                            output_dim=self.user_fea_col['embed_dim'],
                                            embeddings_initializer='random_normal',
                                            embeddings_regularizer=l2(embed_reg))
        self.mlp_item_embedding = Embedding(input_dim=self.item_fea_col['feat_num'],
                                            input_length=1,
                                            output_dim=self.item_fea_col['embed_dim'],
                                            embeddings_initializer='random_normal',
                                            embeddings_regularizer=l2(embed_reg))
        self.dnn = DNN(hidden_units, activation=activation, dnn_dropout=dropout)
        self.dense = Dense(1, activation=None)

    def call(self, inputs):
        user_inputs, pos_inputs, neg_inputs = inputs
        mf_user_embed = self.mf_user_embedding(user_inputs)
        mlp_user_embed = self.mlp_user_embedding(user_inputs)
        mf_pos_embed = self.mf_item_embedding(pos_inputs)
        mf_neg_embed = self.mf_item_embedding(neg_inputs)
        mlp_pos_embed = self.mlp_item_embedding(pos_inputs)
        mlp_neg_embed = self.mlp_item_embedding(neg_inputs)
        mf_pos_vector = tf.nn.sigmoid(tf.multiply(mf_user_embed, mf_pos_embed))
        mf_neg_vector = tf.nn.sigmoid(tf.multiply(mf_user_embed, mf_neg_embed))
        mlp_pos_vector = tf.concat([mlp_user_embed, mlp_pos_embed], axis=-1)
        mlp_neg_vector = tf.concat([tf.tile(mlp_user_embed, multiples=[1, mlp_neg_embed.shape[1], 1]),
                                    mlp_neg_embed], axis=-1)
        mlp_pos_vector = self.dnn(mlp_pos_vector)
        mlp_neg_vector = self.dnn(mlp_neg_vector)
        pos_vector = tf.concat([mf_pos_vector, mlp_pos_vector], axis=-1)
        neg_vector = tf.concat([mf_neg_vector, mlp_neg_vector], axis=-1)
        pos_logits = tf.squeeze(self.dense(pos_vector), axis=-1)
        neg_logits = tf.squeeze(self.dense(neg_vector), axis=-1)
        losses = tf.reduce_mean(- tf.math.log(tf.nn.sigmoid(pos_logits)) -
                                tf.math.log(1 - tf.nn.sigmoid(neg_logits))) / 2
        self.add_loss(losses)
        logits = tf.concat([pos_logits, neg_logits], axis=-1)
        return logits

    def summary(self):
        user_inputs = Input(shape=(1,), dtype=tf.int32)
        pos_inputs = Input(shape=(1,), dtype=tf.int32)
        neg_inputs = Input(shape=(1,), dtype=tf.int32)
        Model(inputs=[user_inputs, pos_inputs, neg_inputs],
              outputs=self.call([user_inputs, pos_inputs, neg_inputs])).summary()

