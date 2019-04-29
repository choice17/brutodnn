import os
locs = ["~/.keras/mnist.npz", "./mnist.npz"]
for loc in locs:
	if os.path.exists(loc):
		print("Dataset exists on %s" % loc)
		exit()

from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
"default directory ~/.keras/mnist.npz"