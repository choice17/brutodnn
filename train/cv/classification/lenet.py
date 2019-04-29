from model.cv.classification.lenet import Lenet
from train.cv.classification.train import Train
from tensorflow.keras.utils import to_categorical
import numpy as np

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

def main():
	dataset = "./data/dataset/mnist.npz"
	data = np.load(dataset)
	batch_size = 128
	num_classes = 10
	epochs = 10
	"""
	Preprocessing
	"""
	x_train, y_train = data.f.x_train, data.f.y_train
	x_test, y_test = data.f.x_test, data.f.y_test
	x_train = np.expand_dims(x_train.astype(np.float32), axis=3)
	x_test = np.expand_dims(x_test.astype(np.float32), axis=3)
	x_train /= 255
	x_test /= 255
	y_train = to_categorical(y_train, num_classes)
	y_test = to_categorical(y_test, num_classes)
	lenet = Lenet.set(include_input=True)
	print(lenet.summary())
	"""
	Training graph
	"""
	sess = tf.Session()
	train = Train(lenet)
	train.buildTrainKeras()
	all_vars = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES)
	saver_tf = tf.train.Saver({v.op.name:v  for v in all_vars})
	history = train.model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
	"""
	Save session instead of .h5 file
	/mtcnn/mtcnn_tf_light/src/mtcnn.py:
	"""
	path_name = "tmp"
	file_name = "lenet"
	saver_tf.save(sess, os.path.join(path_name,file_name), global_step=CURRENT_MINI)
	print(history)
	print()

if __name__ == '__main__':
	main()