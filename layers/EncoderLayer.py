# -*-coding: utf-8-*-
# @Author  : Charlesxu
# @Email   : charlesxu.ai@gmail.com

from layers.FFNLayer import *
from layers.MultiheadAttentionLayer import *

class Encoder(Layer):
    def __init__(self, d_model, num_heads=1, ffn_hidden_unit=128, dropout=0., norm_training=True, causality=True):
        super(Encoder, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads, causality)
        self.ffn = FFN(ffn_hidden_unit, d_model)

        self.layernorm1 = LayerNormalization(epsilon=1e-6, trainable=norm_training)
        self.layernorm2 = LayerNormalization(epsilon=1e-6, trainable=norm_training)

        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def call(self, inputs):
        x, mask = inputs
        att_out = self.mha(x, x, x, mask)
        att_out = self.dropout1(att_out)
        out1 = self.layernorm1(x + att_out)
        ffn_out = self.ffn(out1)
        ffn_out = self.dropout2(ffn_out)
        out2 = self.layernorm2(out1 + ffn_out)
        return out2