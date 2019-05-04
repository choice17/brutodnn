# tensorflow v1.10
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers.advanced_activations import LeakyReLU
from keras.layers.merge import concatenate

import tensorflow as tf
#from tensorflow.keras.metrics import categorical_accuracy, top_k_categorical_accuracy, categorical_crossentropy
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
#from tensorflow.keras.optimizers import Adam
#from tensorflow.keras.applications import MobileNet

IMAGE_H, IMAGE_W = 416, 416

def space_to_depth_x2(x):
    return tf.space_to_depth(x, block_size=2)

class Yolov2_Base(object):

	def set(inputs, include_input=False):
		"""
		input should be HxWx3 input
		Default shape 416x416x3
		"""
		if include_input:
			input_image = Input(shape=(IMAGE_H, IMAGE_W, 3), name='input')
			#true_boxes  = Input(shape=(1, 1, 1, TRUE_BOX_BUFFER , 4))
		idx = 0
		conv_n = 'conv_%d'
		norm_n = 'norm_%d'

		idx += 1
		# Layer 1 (Dim:416x416x3, /1)
		x = Conv2D(32, (3,3), strides=(1,1), padding='same', name=conv_n%idx, use_bias=False)(input_image)
		x = BatchNormalization(name=norm_n%idx)(x)
		x = LeakyReLU(alpha=0.1)(x)
		x = MaxPooling2D(pool_size=(2, 2))(x)

		idx += 1
		# Layer 2 (Dim:208x208x32, /2)
		x = Conv2D(64, (3,3), strides=(1,1), padding='same', name=conv_n%idx, use_bias=False)(x)
		x = BatchNormalization(name=norm_n%idx)(x)
		x = LeakyReLU(alpha=0.1)(x)
		x = MaxPooling2D(pool_size=(2, 2))(x)

		idx += 1
		# Layer 3 (Dim:104x104x64, /4)
		x = Conv2D(128, (3,3), strides=(1,1), padding='same', name=conv_n%idx, use_bias=False)(x)
		x = BatchNormalization(name=norm_n%idx)(x)
		x = LeakyReLU(alpha=0.1)(x)

		idx += 1
		# Layer 4 (Dim:104x104x128, /4)
		x = Conv2D(64, (1,1), strides=(1,1), padding='same', name=conv_n%idx, use_bias=False)(x)
		x = BatchNormalization(name=norm_n%idx)(x)
		x = LeakyReLU(alpha=0.1)(x)

		idx += 1
		# Layer 5 (Dim:104x104x64, /4)
		x = Conv2D(128, (3,3), strides=(1,1), padding='same', name=conv_n%idx, use_bias=False)(x)
		x = BatchNormalization(name=norm_n%idx)(x)
		x = LeakyReLU(alpha=0.1)(x)
		x = MaxPooling2D(pool_size=(2, 2))(x)

		idx += 1
		# Layer 6 (Dim:52x52x128, /8)
		x = Conv2D(256, (3,3), strides=(1,1), padding='same', name=conv_n%idx, use_bias=False)(x)
		x = BatchNormalization(name=norm_n%idx)(x)
		x = LeakyReLU(alpha=0.1)(x)

		idx += 1
		# Layer 7 (Dim:52x52x256, /8)
		x = Conv2D(128, (1,1), strides=(1,1), padding='same', name=conv_n%idx, use_bias=False)(x)
		x = BatchNormalization(name=norm_n%idx)(x)
		x = LeakyReLU(alpha=0.1)(x)

		idx += 1
		# Layer 8 (Dim:52x52x128, /8)
		x = Conv2D(256, (3,3), strides=(1,1), padding='same', name=conv_n%idx, use_bias=False, input_shape=(416,416,3))(x)
		x = BatchNormalization(name=norm_n%idx)(x)
		x = LeakyReLU(alpha=0.1)(x)
		x = MaxPooling2D(pool_size=(2, 2))(x)

		idx += 1
		# Layer 9 (Dim:26x26x256, /16)
		x = Conv2D(512, (3,3), strides=(1,1), padding='same', name=conv_n%idx, use_bias=False)(x)
		x = BatchNormalization(name=norm_n%idx)(x)
		x = LeakyReLU(alpha=0.1)(x)

		idx += 1
		# Layer 10 (Dim:26x26x512, /16)
		x = Conv2D(256, (1,1), strides=(1,1), padding='same', name=conv_n%idx, use_bias=False)(x)
		x = BatchNormalization(name=norm_n%idx)(x)
		x = LeakyReLU(alpha=0.1)(x)

		idx += 1
		# Layer 11 (Dim:26x26x128, /16)
		x = Conv2D(512, (3,3), strides=(1,1), padding='same', name=conv_n%idx, use_bias=False)(x)
		x = BatchNormalization(name=norm_n%idx)(x)
		x = LeakyReLU(alpha=0.1)(x)

		idx += 1
		# Layer 12 (Dim:26x26x512, /16)
		x = Conv2D(256, (1,1), strides=(1,1), padding='same', name=conv_n%idx, use_bias=False)(x)
		x = BatchNormalization(name=norm_n%idx)(x)
		x = LeakyReLU(alpha=0.1)(x)

		idx += 1
		# Layer 13 (Dim:26x26x256, /16)
		x = Conv2D(512, (3,3), strides=(1,1), padding='same', name=conv_n%idx, use_bias=False)(x)
		x = BatchNormalization(name=norm_n%idx)(x)
		x = LeakyReLU(alpha=0.1)(x)

		skip_connection = x

		x = MaxPooling2D(pool_size=(2, 2))(x)

		idx += 1
		# Layer 14 (Dim:13x13x512, /32)
		x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name=conv_n%idx, use_bias=False)(x)
		x = BatchNormalization(name=norm_n%idx)(x)
		x = LeakyReLU(alpha=0.1)(x)

		idx += 1
		# Layer 15 (Dim:13x13x1024, /32)
		x = Conv2D(512, (1,1), strides=(1,1), padding='same', name=conv_n%idx, use_bias=False)(x)
		x = BatchNormalization(name=norm_n%idx)(x)
		x = LeakyReLU(alpha=0.1)(x)

		idx += 1
		# Layer 16 (Dim:13x13x512, /32)
		x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name=conv_n%idx, use_bias=False)(x)
		x = BatchNormalization(name=norm_n%idx)(x)
		x = LeakyReLU(alpha=0.1)(x)

		idx += 1
		# Layer 17 (Dim:13x13x1024, /32)
		x = Conv2D(512, (1,1), strides=(1,1), padding='same', name=conv_n%idx, use_bias=False)(x)
		x = BatchNormalization(name=norm_n%idx)(x)
		x = LeakyReLU(alpha=0.1)(x)

		idx += 1
		# Layer 18 (Dim:13x13x512, /32)
		x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name=conv_n%idx, use_bias=False)(x)
		x = BatchNormalization(name=norm_n%idx)(x)
		x = LeakyReLU(alpha=0.1)(x)

		return x, skip_connection, idx

