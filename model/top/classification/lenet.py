# tensorflow v1.10
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation



class Lenet_Top(object):

	def set(inputs):
		"""
		input should be XxX input

		"""
		x = Flatten()(inputs)
		x = Dense(units=120, activation="relu")(x)
		x = Dense(units=84, activation='relu')(x)
		return x