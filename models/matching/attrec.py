# -*-coding: utf-8-*-
# @Author  : Charlesxu
# @Email   : charlesxu.ai@gmail.com

from models.matching import *
from layers.SelfAttentionLayer import *

class AttRec(Model):
    def __init__(self, feature_columns, maxlen=40, mode='inner', gamma=0.5, w=0.5, embed_reg=1e-6, **kwargs):
        super(AttRec, self).__init__(**kwargs)
        self.maxlen = maxlen
        self.w = w
        self.gamma = gamma
        self.mode = mode
        self.user_fea_col, self.item_fea_col = feature_columns
        self.embed_dim = self.item_fea_col['embed_dim']
        self.user_embedding = Embedding(input_dim=self.user_fea_col['feat_num'],
                                        input_length=1,
                                        output_dim=self.user_fea_col['embed_dim'],
                                        mask_zero=False,
                                        embeddings_initializer='random_normal',
                                        embeddings_regularizer=l2(embed_reg))
        self.item_embedding = Embedding(input_dim=self.item_fea_col['feat_num'],
                                        input_length=1,
                                        output_dim=self.item_fea_col['embed_dim'],
                                        mask_zero=True,
                                        embeddings_initializer='random_normal',
                                        embeddings_regularizer=l2(embed_reg))
        self.item2_embedding = Embedding(input_dim=self.item_fea_col['feat_num'],
                                        input_length=1,
                                        output_dim=self.item_fea_col['embed_dim'],
                                        mask_zero=True,
                                        embeddings_initializer='random_normal',
                                        embeddings_regularizer=l2(embed_reg))
        self.self_attention = SelfAttention()

    def call(self, inputs, **kwargs):
        user_inputs, seq_inputs, pos_inputs, neg_inputs = inputs
        mask = tf.cast(tf.not_equal(seq_inputs, 0), dtype=tf.float32)
        user_embed = self.user_embedding(tf.squeeze(user_inputs, axis=-1))
        seq_embed = self.item_embedding(seq_inputs)
        pos_embed = self.item_embedding(tf.squeeze(pos_inputs, axis=-1))
        neg_embed = self.item_embedding(tf.squeeze(neg_inputs, axis=-1))
        pos_embed2 = self.item2_embedding(tf.squeeze(pos_inputs, axis=-1))
        neg_embed2 = self.item2_embedding(tf.squeeze(neg_inputs, axis=-1))
        short_interest = self.self_attention([seq_embed, seq_embed, seq_embed, mask])
        if self.mode == 'inner':
            pos_long_interest = tf.multiply(user_embed, pos_embed2)
            neg_long_interest = tf.multiply(user_embed, neg_embed2)
            pos_scores = self.w * tf.reduce_sum(pos_long_interest, axis=-1, keepdims=True) \
                         + (1 - self.w) * tf.reduce_sum(tf.multiply(short_interest, pos_embed), axis=-1, keepdims=True)
            neg_scores = self.w * tf.reduce_sum(neg_long_interest, axis=-1, keepdims=True) \
                         + (1 - self.w) * tf.reduce_sum(tf.multiply(short_interest, neg_embed), axis=-1, keepdims=True)
            self.add_loss(tf.reduce_mean(-tf.math.log(tf.nn.sigmoid(pos_scores - neg_scores))))
        else:
            user_embed = tf.clip_by_norm(user_embed, 1, -1)
            pos_embed = tf.clip_by_norm(pos_embed, 1, -1)
            neg_embed = tf.clip_by_norm(neg_embed, 1, -1)
            pos_embed2 = tf.clip_by_norm(pos_embed2, 1, -1)
            neg_embed2 = tf.clip_by_norm(neg_embed2, 1, -1)
            pos_long_interest = tf.square(user_embed - pos_embed2)
            neg_long_interest = tf.square(user_embed - neg_embed2)
            pos_scores = self.w * tf.reduce_sum(pos_long_interest, axis=-1, keepdims=True) + \
                         (1 - self.w) * tf.reduce_sum(tf.square(short_interest - pos_embed), axis=-1, keepdims=True)
            neg_scores = self.w * tf.reduce_sum(neg_long_interest, axis=-1, keepdims=True) + \
                         (1 - self.w) * tf.reduce_sum(tf.square(short_interest - neg_embed), axis=-1, keepdims=True)
            self.add_loss(tf.reduce_sum(tf.nn.relu(pos_scores - neg_scores + self.gamma)))
        return pos_scores, neg_scores

    def summary(self):
        seq_inputs = Input(shape=(self.maxlen,), dtype=tf.int32)
        user_inputs = Input(shape=(1, ), dtype=tf.int32)
        pos_inputs = Input(shape=(1, ), dtype=tf.int32)
        neg_inputs = Input(shape=(1, ), dtype=tf.int32)
        Model(inputs=[user_inputs, seq_inputs, pos_inputs, neg_inputs],
            outputs=self.call([user_inputs, seq_inputs, pos_inputs, neg_inputs])).summary()