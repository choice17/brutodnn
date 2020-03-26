import numpy as np
from tensorflow.keras.layers import Conv2D, Input, Dense
from tensorflow.keras.layers import MaxPooling2D, Dropout, Flatten


INPUT_DIM = (224, 224, 3)
NUM_CLASSES = 1000

class Vgg16(object):

	def _conv_block(x, convs, maxpool=True):
		for conv in convs:
			x = Conv2D(conv['filter'],
					   conv['kernel'],
					   strides=conv['strides'],
					   padding='same',
					   activation='relu')(x)
		if maxpool:
			x = MaxPooling2D(pool_size=(2, 2))(x)
		return x

	def getBase(inputs):

		x = Vgg16._conv_block(inputs, [{'filter':64, 'kernel':(3, 3), 'strides':(1, 1)},
									   {'filter':64, 'kernel':(3, 3), 'strides':(1, 1)}], maxpool=True)

		x = Vgg16._conv_block(inputs, [{'filter':128, 'kernel':(3, 3), 'strides':(1, 1)},
									   {'filter':128, 'kernel':(3, 3), 'strides':(1, 1)}], maxpool=True)

		x = Vgg16._conv_block(inputs, [{'filter':256, 'kernel':(3, 3), 'strides':(1, 1)},
									   {'filter':256, 'kernel':(3, 3), 'strides':(1, 1)},
									   {'filter':256, 'kernel':(3, 3), 'strides':(1, 1)}], maxpool=True)

		x = Vgg16._conv_block(inputs, [{'filter':512, 'kernel':(3, 3), 'strides':(1, 1)},
									   {'filter':512, 'kernel':(3, 3), 'strides':(1, 1)},
									   {'filter':512, 'kernel':(3, 3), 'strides':(1, 1)}], maxpool=True)

		x = Vgg16._conv_block(inputs, [{'filter':512, 'kernel':(3, 3), 'strides':(1, 1)},
									   {'filter':512, 'kernel':(3, 3), 'strides':(1, 1)},
									   {'filter':512, 'kernel':(3, 3), 'strides':(1, 1)}], maxpool=True)

		return x

	def getTop(inputs):
		x = Flatten()(inputs)
		x = Dense(4096,activation='relu')(x)
		x = Dropout(0.5)(x)
		x = Dense(4096,activation='relu')(x)
		x = Dropout(0.5)(x)
		return x

	def set(inputs=None, include_input=False, num_class=NUM_CLASSES, output='softmax', input_dim=INPUT_DIM):
		if include_input:
			inputs = Input(shape=input_dim)
		x = Vgg16.getBase(inputs)
		x = Vgg16.getTop(x)
		x = Dense(num_class, activation=output)(x)
		model = Model(inputs, x)
		model.summary()
		return model


