import numpy as np
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Input, Flatten
from tensorflow.keras.models import Model

from tensorflow.keras import backend as K
from tensorflow.keras.applications import AlexNet

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
        x = Dropout(0.5)(x)
        x = Dense(4096,activation='relu')(x)
        x = Dropout(0.5)(x)
        return x

    def set(inputs=None, include_inputs=False, class_num=CLASS_NUM, output='softmax', input_dim=INPUT_DIM):
        if include_inputs:
            inputs = Input(shape=INPUT_DIM)
        x = Alexnet.getBase(inputs)
        x = Alexnet.getTop(x)
        x = Dense(class_num, activation=output)(x)
        model = Model(inputs, x)
        return model

    def getKerasModelBase(num_class=CLASS_NUM, output='softmax', fix_layer=6):
        base_model=AlexNet(weights='imagenet',include_top=False) #imports the mobilenet model and discards the last 1000 neuron layer.
        x=base_model.output
        x = Dense(num_class, activation=output)(x)
        outputs = Reshape((num_class,))(x)
        model=Model(inputs=base_model.input,outputs=outputs)
        for layer in model.layers:
            layer.trainable=False
        # or if we want to set the first 20 layers of the network to be non-trainable
        for layer in model.layers[:fix_layer]:
            layer.trainable=False
        for layer in model.layers[fix_layer:]:
            layer.trainable=True
        return model