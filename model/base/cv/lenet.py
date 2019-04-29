# tensorflow v1.10
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation, Input
#from tensorflow.keras.metrics import categorical_accuracy, top_k_categorical_accuracy, categorical_crossentropy
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
#from tensorflow.keras.optimizers import Adam
#from tensorflow.keras.applications import MobileNet


class Lenet_Base(object):

	def set(inputs, include_input=False):
		"""
		input should be XxX input

		"""
		if include_input:
			inputs = Input(shape=(28, 28, 1))
		x = Conv2D(filters=6, kernel_size=(3, 3), activation="relu", input_shape=(32, 32, 1))(inputs)
		x = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(x)
		return x		

