import numpy as np
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Input, Flatten
from tensorflow.keras.models import Model

from tensorflow.keras import backend as K


INPUT_DIM = (227, 227, 3)
CLASS_NUM = 1000

#https://blog.csdn.net/wmy199216/article/details/71171401

class Alexnet(object):

    def getBase(inputs, include_inputs=False):

        if include_inputs:
            inputs = Input(shape=INPUT_DIM)

        x = Conv2D(96, (11, 11), strides=(4, 4), padding='valid', activation='relu', kernel_initializer='uniform')(inputs)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
        x = Conv2D(256, (5, 5), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
        x = Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x)
        x = Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x)
        x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform')(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
        return x

    def getTop(inputs):
        x = Flatten()(inputs)
        x = Dense(4096,activation='relu')(x)
        x = Dropout(rete=0.3)(x)
        x = Dense(4096,activation='relu')(x)
        x = Dropout(rate=0.3)(x)
        return x

    def set(inputs=None, include_inputs=False, class_num=CLASS_NUM, output='softmax', input_dim=INPUT_DIM):
        if include_inputs:
            inputs = Input(shape=INPUT_DIM)
        x = Alexnet.getBase(inputs)
        x = Alexnet.getTop(x)
        x = Dense(class_num, activation=output)(x)
        model = Model(inputs, x)
        return model

