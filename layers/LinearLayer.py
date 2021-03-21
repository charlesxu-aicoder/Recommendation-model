# -*-coding: utf-8-*-
# @Author  : Charlesxu
# @Email   : charlesxu.ai@gmail.com

from layers import *

class Linear(layers.Layer):
    def __init__(self):
        super(Linear, self).__init__()
        self.dense = Dense(1, activation=None)

    def call(self, inputs, **kwargs):
        result = self.dense(inputs)
        return result