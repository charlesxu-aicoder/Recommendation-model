# -*-coding: utf-8-*-
# @Author  : Charlesxu
# @Email   : charlesxu.ai@gmail.com

from layers import *

class DNN(layers.Layer):
    def __init__(self, hidden_units, dnn_dropout=0., dnn_activation='relu'):
        super(DNN, self).__init__()
        self.dnn_network = [Dense(units=unit, activation=dnn_activation) for unit in hidden_units]
        self.dropout = Dropout(dnn_dropout)

    def call(self, inputs, **kwargs):
        x = inputs
        for dnn in self.dnn_network:
            x = dnn(x)
        x = self.dropout(x)
        return x