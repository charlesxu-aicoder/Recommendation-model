# -*-coding: utf-8-*-
# @Author  : Charlesxu
# @Email   : charlesxu.ai@gmail.com

from layers import *

class FFN(Layer):
    def __init__(self, hidden_unit, d_model):
        """
        Feed Forward Network
        :param hidden_unit: A scalar. W1
        :param d_model: A scalar. W2
        """
        super(FFN, self).__init__()
        self.conv1 = Conv1D(filters=hidden_unit, kernel_size=1, activation='relu', use_bias=True)
        self.conv2 = Conv1D(filters=d_model, kernel_size=1, activation=None, use_bias=True)

    def call(self, inputs):
        x = self.conv1(inputs)
        output = self.conv2(x)
        return output