# -*-coding: utf-8-*-
# @Author  : Charlesxu
# @Email   : charlesxu.ai@gmail.com

import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model,layers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Embedding, Dense, Input, Layer, Dropout, Flatten,ReLU,LayerNormalization,BatchNormalization,PReLU
from tensorflow.keras.utils import plot_model
