# -*-coding: utf-8-*-
# @Author  : Charlesxu
# @Email   : charlesxu.ai@gmail.com

from layers.EncoderLayer import *
from models.matching import *

class SASRec(tf.keras.Model):
    def __init__(self, item_fea_col, blocks=1, num_heads=1, ffn_hidden_unit=128,
                 dropout=0., maxlen=40, norm_training=True, causality=False, embed_reg=1e-6):
        super(SASRec, self).__init__()
        self.maxlen = maxlen
        self.item_fea_col = item_fea_col
        self.embed_dim = self.item_fea_col['embed_dim']
        self.d_model = self.embed_dim
        self.item_embedding = Embedding(input_dim=self.item_fea_col['feat_num'],
                                        input_length=1,
                                        output_dim=self.item_fea_col['embed_dim'],
                                        mask_zero=True,
                                        embeddings_initializer='random_uniform',
                                        embeddings_regularizer=l2(embed_reg))
        self.pos_embedding = Embedding(input_dim=self.maxlen,
                                       input_length=1,
                                       output_dim=self.embed_dim,
                                       mask_zero=False,
                                       embeddings_initializer='random_uniform',
                                       embeddings_regularizer=l2(embed_reg))
        self.dropout = Dropout(dropout)
        self.encoder_layer = [Encoder(self.d_model, num_heads, ffn_hidden_unit,
                                           dropout, norm_training, causality) for b in range(blocks)]

    def call(self, inputs, training=None):
        seq_inputs, pos_inputs, neg_inputs = inputs
        mask = tf.expand_dims(tf.cast(tf.not_equal(seq_inputs, 0), dtype=tf.float32), axis=-1)
        seq_embed = self.item_embedding(seq_inputs)
        pos_encoding = tf.expand_dims(self.pos_embedding(tf.range(self.maxlen)), axis=0)
        seq_embed += pos_encoding
        seq_embed = self.dropout(seq_embed)
        att_outputs = seq_embed
        att_outputs *= mask

        for block in self.encoder_layer:
            att_outputs = block([att_outputs, mask])
            att_outputs *= mask

        user_info = tf.expand_dims(att_outputs[:, -1], axis=1)
        pos_info = self.item_embedding(pos_inputs)
        neg_info = self.item_embedding(neg_inputs)
        pos_logits = tf.reduce_sum(user_info * pos_info, axis=-1)
        neg_logits = tf.reduce_sum(user_info * neg_info, axis=-1)
        losses = tf.reduce_mean(- tf.math.log(tf.nn.sigmoid(pos_logits)) -
                                tf.math.log(1 - tf.nn.sigmoid(neg_logits))) / 2
        self.add_loss(losses)
        logits = tf.concat([pos_logits, neg_logits], axis=-1)
        return logits

    def summary(self):
        seq_inputs = Input(shape=(self.maxlen,), dtype=tf.int32)
        pos_inputs = Input(shape=(1,), dtype=tf.int32)
        neg_inputs = Input(shape=(1,), dtype=tf.int32)
        tf.keras.Model(inputs=[seq_inputs, pos_inputs, neg_inputs],
                       outputs=self.call([seq_inputs, pos_inputs, neg_inputs])).summary()

