import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dropout, ReLU
from tensorflow.keras.layers import Activation, BatchNormalization, add, Reshape, DepthwiseConv2D
#from tensorflow.keras.applications.mobilenet import relu6, DepthwiseConv2D
#from keras.utils.vis_utils import plot_model
from tensorflow.keras import backend as K
from tensorflow.keras.applications import MobileNetV2

INPUT_DIM = (224, 224, 3)
CLASS_NUM = 1000

#https://blog.csdn.net/wmy199216/article/details/71171401
"""
def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:-o
        new_v += divisor
    return new_v
"""

class Mobilenetv2(object):

    def relu6(*args, **kwargs):
        return ReLU(6., *args, **kwargs)

    def _conv_block(inputs, filters, kernel, strides):
        """Convolution Block
        This function defines a 2D convolution operation with BN and relu6.
        # Arguments
            inputs: Tensor, input tensor of conv layer.
            filters: Integer, the dimensionality of the output space.
            kernel: An integer or tuple/list of 2 integers, specifying the
                width and height of the 2D convolution window.
            strides: An integer or tuple/list of 2 integers,
                specifying the strides of the convolution along the width and height.
                Can be a single integer to specify the same value for
                all spatial dimensions.
        # Returns
            Output tensor.
        """

        channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

        x = Conv2D(filters, kernel, padding='same', strides=strides)(inputs)
        x = BatchNormalization(axis=channel_axis)(x)
        return Mobilenetv2.relu6()(x)


    def _bottleneck(inputs, filters, kernel, t, s, r=False):
        """Bottleneck
        This function defines a basic bottleneck structure.
        # Arguments
            inputs: Tensor, input tensor of conv layer.
            filters: Integer, the dimensionality of the output space.
            kernel: An integer or tuple/list of 2 integers, specifying the
                width and height of the 2D convolution window.
            t: Integer, expansion factor.
                t is always applied to the input size.
            s: An integer or tuple/list of 2 integers,specifying the strides
                of the convolution along the width and height.Can be a single
                integer to specify the same value for all spatial dimensions.
            r: Boolean, Whether to use the residuals.
        # Returns
            Output tensor.
        """

        channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
        tchannel = K.int_shape(inputs)[channel_axis] * t

        x = Mobilenetv2._conv_block(inputs, tchannel, (1, 1), (1, 1))

        x = DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same')(x)
        x = BatchNormalization(axis=channel_axis)(x)
        x = Mobilenetv2.relu6()(x)

        x = Conv2D(filters, (1, 1), strides=(1, 1), padding='same')(x)
        x = BatchNormalization(axis=channel_axis)(x)

        if r:
            x = add([x, inputs])
        return x

    def _inverted_residual_block(inputs, filters, kernel, t, strides, n):
        """Inverted Residual Block
        This function defines a sequence of 1 or more identical layers.
        # Arguments
            inputs: Tensor, input tensor of conv layer.
            filters: Integer, the dimensionality of the output space.
            kernel: An integer or tuple/list of 2 integers, specifying the
                width and height of the 2D convolution window.
            t: Integer, expansion factor.
                t is always applied to the input size.
            s: An integer or tuple/list of 2 integers,specifying the strides
                of the convolution along the width and height.Can be a single
                integer to specify the same value for all spatial dimensions.
            n: Integer, layer repeat times.
        # Returns
            Output tensor.
        """

        x = Mobilenetv2._bottleneck(inputs, filters, kernel, t, strides)

        for i in range(1, n):
            x = Mobilenetv2._bottleneck(x, filters, kernel, t, 1, True)

        return x

    def getBase(inputs, include_inputs=False, alpha= 1.0):

        if include_inputs:
            inputs = Input(shape=input_shape)

        x = Mobilenetv2._conv_block(inputs, 32, (3, 3), strides=(2, 2))

        x = Mobilenetv2._inverted_residual_block(x, 16, (3, 3), t=1, strides=1, n=1)
        x = Mobilenetv2._inverted_residual_block(x, 24, (3, 3), t=6, strides=2, n=2)
        x = Mobilenetv2._inverted_residual_block(x, 32, (3, 3), t=6, strides=2, n=3)
        x = Mobilenetv2._inverted_residual_block(x, 64, (3, 3), t=6, strides=2, n=4)
        x = Mobilenetv2._inverted_residual_block(x, 96, (3, 3), t=6, strides=1, n=3)
        x = Mobilenetv2._inverted_residual_block(x, 160, (3, 3), t=6, strides=2, n=3)
        x = Mobilenetv2._inverted_residual_block(x, 320, (3, 3), t=6, strides=1, n=1)

        x = Mobilenetv2._conv_block(x, 1280, (1, 1), strides=(1, 1))

        return x


    def getTop(inputs):
        x = GlobalAveragePooling2D()(inputs)
        x = Reshape((1, 1, 1280))(x)
        x = Dropout(0.3, name='Dropout')(x)
        return x

    def set(inputs=None, include_inputs=False, class_num=CLASS_NUM, output='softmax', input_dim=INPUT_DIM):

        if include_inputs:
            inputs = Input(shape=INPUT_DIM)
        base = Mobilenetv2.getBase(inputs)
        top = Mobilenetv2.getTop(base)
        x = Conv2D(class_num, (1, 1), padding='same')(top)
        x = Activation(output)(x)
        outputs = Reshape((class_num,))(x)
        model = Model(inputs, outputs)
        #plot_model(model, to_file='images/MobileNetv2.png', show_shapes=True)
        return model

    def getKerasModel(*args, **kwargs):
        return MobileNetV2(*args, **kwargs)

    def getKerasModelBase(num_class=CLASS_NUM, output='softmax', fix_layer=20):
        base_model=MobileNetV2(weights='imagenet',include_top=False) #imports the mobilenet model and discards the last 1000 neuron layer.
        x=base_model.output
        x=GlobalAveragePooling2D()(x)
        x = Reshape((1, 1, 1280))(x)
        #x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
        #x=Dense(1024,activation='relu')(x) #dense layer 2
        #x=Dense(512,activation='relu')(x) #dense layer 3
        #preds=Dense(num_class,activation=output)(x) #final layer with softmax activation
        x = Conv2D(1024, (1, 1), padding='same')(x)
        x = Mobilenetv2.relu6()(x)
        x = Conv2D(1024, (1, 1), padding='same')(x)
        x = Mobilenetv2.relu6()(x)
        x = Conv2D(num_class, (1, 1), padding='same')(x)
        x = Activation(output)(x)
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


