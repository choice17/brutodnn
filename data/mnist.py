import os
import numpy as np
from tensorflow.keras.utils import to_categorical
path = os.path.join("data","dataset","mnist.npz") 

class MNIST(object):

    num_classes = 10
    path = path

    def __init__(self):
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.x_valid = None
        self.y_valid = None
        self.batch_size = 0
        self.current_step = 0
        self.steps_per_epoch = 0
        self.x_train_idx = None
        self.x_valid_idx = None

    def load(self, path=path):
        data = np.load(path)
        self.x_train, self.y_train = data.f.x_train, data.f.y_train
        self.x_test, self.y_test = data.f.x_test, data.f.y_test

    def preprocess_image(x):
        """
        x : 1x28x28x1
        pixel : 0 - 255
        """
        return x / 255


    def preprocess_dataset(self):
        x_train = np.expand_dims(self.x_train.astype(np.float32), axis=3)
        x_test = np.expand_dims(self.x_test.astype(np.float32), axis=3)
        self.x_train = x_train / 255
        self.x_test = x_test / 255
        self.convert_toCategorical()

    def convert_toCategorical(self):
        self.y_train = to_categorical(self.y_train, self.num_classes)
        self.y_test = to_categorical(self.y_test, self.num_classes)

    def shuffle(self, num, onData=False):
        self.shuffle_idx = np.arange(num)
        self.random.shuffle(self.shuffle_idx)

    def split_train_valid(self, ratio=0.9):
        _len = int(self.x_train.shape[0] * ratio)
        self.train_idx = self.shuffle_idx[:_len]
        self.valid_idx = self.shuffle_idx[_len:]

    def setConfig(self, batch_size):
        if self.x_valid_idx is not None:
            _len = len(self.train_idx)
        else:
            _len = self.x_train.shape[0]
        self.batch_size = batch_size
        self.steps_per_epoch = int(_len / batch_size)

    def shuffle_on_train(self):
        self.random.shuffle(self.x_train_idx)

    def getNext(self):
        if self.current_step > self.steps_per_epoch:
           self.current_step = 0
           self.shuffle_on_train()
        idx = self.current_step * self.batch_size
        idx_to = idx+self.batch_size
        _len = len(self.train_idx)
        if idx_to > _len:
           idx_to = _len
        train_idx = self.train_idx[idx: idx_to]
        x_train_batch = self.x_train[train_idx, ...]
        y_train_batch = self.y_train[train_idx, ...]
        self.current_step += 1
        return x_train_batch, y_train_batch

    def getValid(self):
        if self.valid_idx:
            x_valid = self.x_train[self.valid_idx]
            y_valid = self.y_train[self.valid_idx]
            return x_valid, y_valid
        else:
            return self.x_test, self.y_test



