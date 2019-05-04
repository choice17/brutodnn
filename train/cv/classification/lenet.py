from model.cv.classification.lenet import Lenet
from train.cv.classification.train import Train
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
import tensorflow as tf
import numpy as np
import os
from tensorflow.python.client import device_lib
from data.mnist import MNIST

print(device_lib.list_local_devices())

batch_size = 128
num_classes = MNIST.num_classes
epochs = 1
dataset = MNIST.path
path_name = "tmp"
file_name = "lenet.pb"

if not os.path.exists(path_name): os.mkdir(path_name)

class Lenet_Train(Train):
    def __init__(self):
        pass
    def getData(self):
        mnist = MNIST()
        mnist.load()
        mnist.preprocess_dataset()

        self.x_train = mnist.x_train
        self.x_test = mnist.x_test
        self.y_train = mnist.y_train
        self.y_test = mnist.y_test

    def initModel(self):
        self.model = Lenet.set(
            include_input=True,
            class_num=MNIST.num_classes,
            output='sigmoid')

    def getTfSaver(self):
        all_vars = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES)
        self.saver_tf = tf.train.Saver({v.op.name:v  for v in all_vars})

    def fit(self):
        self.getTfSaver()
        early_stop = EarlyStopping(monitor='val_loss', 
                           min_delta=0.001, 
                           patience=3, 
                           mode='min', 
                           verbose=1)

        checkpoint = ModelCheckpoint('mnist.h5', 
                                    monitor='val_loss', 
                                    verbose=1, 
                                    save_best_only=True, 
                                    mode='min', 
                                    period=1)
        tb_counter  = 1
        tensorboard = TensorBoard(log_dir=os.path.expanduser('~/logs/') + 'mnist' + '_' + str(tb_counter), 
                                histogram_freq=0, 
                                write_graph=True, 
                                write_images=False)

        self.history = self.model.fit(self.x_train, self.y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        callbacks        = [early_stop, checkpoint, tensorboard], 
                        validation_data=(self.x_test, self.y_test))
    
    def save(self):
        frozen_graph = Train.freeze_session(K.get_session(),
                              output_names=[out.op.name for out in self.model.outputs])
        tf.train.write_graph(frozen_graph, path_name, file_name, as_text=False)
        #self.saver_tf.save(self.sess, os.path.join(path_name,file_name), global_step=999)

    def evaluate(self):
        score = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

    def buildTrainKeras(self):
        optimizer = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        #optimizer = SGD(lr=1e-4, decay=0.0005, momentum=0.9)
        #optimizer = RMSprop(lr=1e-5, rho=0.9, epsilon=1e-08, decay=0.0)
        self.model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
        #model.compile(loss=custom_loss, optimizer=optimizer)


    def run(self):
        self.getData()
        self.initModel()
        self.getTfSaver()
        self.buildTrainKeras()
        self.fit()
        self.save()
        print()

def main():
    train = Lenet_Train()
    train.run()
    """
    dataset = "./data/dataset/mnist.npz"
    data = np.load(dataset)
    batch_size = 128
    num_classes = 10
    epochs = 1
    
    #Preprocessing

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
    
    #Training graph
    
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
    
    #Save session instead of .h5 file
    #/mtcnn/mtcnn_tf_light/src/mtcnn.py:

    path_name = "tmp"
    file_name = "lenet"
    saver_tf.save(sess, os.path.join(path_name,file_name), global_step=CURRENT_MINI)
    print(history)
    print()
    """
if __name__ == '__main__':
    main()