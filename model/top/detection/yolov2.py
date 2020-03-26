# tensorflow v1.10
from tensorflow import keras
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers.advanced_activations import LeakyReLU
from keras.layers.merge import concatenate

import tensorflow as tf

def space_to_depth_x2(x):
    return tf.space_to_depth(x, block_size=2)

class Yolov2_Top(object):

	### Yolov2 top ### 
	def set(inputs, idx=19):
		assert type(inputs) == list, "Not correct input type"
		assert len(inputs) == 2, "Not correct input size"
		x = inputs[0]
		skip_connection = inputs[1]

		conv_n = 'conv_%d'
		norm_n = 'norm_%d'

		# Layer 19 (Dim:13x13x1024, /32)
		x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name=conv_n%idx, use_bias=False)(x)
		x = BatchNormalization(name=norm_n%idx)(x)
		x = LeakyReLU(alpha=0.1)(x)

		idx += 1
		# Layer 20 (Dim:13x13x1024, /32)
		x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name=conv_n%idx, use_bias=False)(x)
		x = BatchNormalization(name=norm_n%idx)(x)
		x = LeakyReLU(alpha=0.1)(x)

		idx += 1
		# Layer 21 (Dim:26x26x512, /16)
		skip_connection = Conv2D(64, (1,1), strides=(1,1), padding='same', name=conv_n%idx, use_bias=False)(skip_connection)
		skip_connection = BatchNormalization(name=norm_n%idx)(skip_connection)
		skip_connection = LeakyReLU(alpha=0.1)(skip_connection)
		skip_connection = Lambda(space_to_depth_x2)(skip_connection)

		x = concatenate([skip_connection, x])

		idx += 1
		# Layer 22 (Dim:13x13x2560, /32)
		x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name=conv_n%%idx, use_bias=False)(x)
		x = BatchNormalization(name=norm_n%idx)(x)
		x = LeakyReLU(alpha=0.1)(x)

		""" Here is the default output
		# Layer 23 (Dim:13x13x1024, /32)
		x = Conv2D(BOX * (4 + 1 + CLASS), (1,1), strides=(1,1), padding='same', name='conv_23')(x)
		output = Reshape((GRID_H, GRID_W, BOX, 4 + 1 + CLASS))(x)

		# small hack to allow true_boxes to be registered when Keras build the model 
		# for more information: https://github.com/fchollet/keras/issues/2790
		output = Lambda(lambda args: args[0])([output, true_boxes])

		model = Model([input_image, true_boxes], output)
		"""

	return x, idx