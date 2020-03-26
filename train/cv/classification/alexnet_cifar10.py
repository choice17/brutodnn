import keras
from keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint

import numpy as np
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Input, Flatten
from tensorflow.keras.models import Model

from tensorflow.keras import backend as K


INPUT_DIM = (227, 227, 3)
CLASS_NUM = 1000

#https://blog.csdn.net/wmy199216/article/details/71171401

class Alexnet(object):

    def getBase(inputs, include_inputs=False):

        if include_inputs:
            inputs = Input(shape=INPUT_DIM)

        x = Conv2D(96, (7, 7), strides=(2, 2), padding='valid', activation='relu')(inputs)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
        x = Conv2D(256, (5, 5), strides=(1, 1), padding='same', activation='relu')(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
        x = Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
        x = Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
        x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
        return x

    def getTop(inputs):
        x = Flatten()(inputs)
        x = Dense(4096,activation='relu')(x)
        x = Dropout(rate=0.3)(x)
        x = Dense(4096,activation='relu')(x)
        x = Dropout(rate=0.3)(x)
        return x

    def set(inputs=None, include_inputs=False, class_num=CLASS_NUM, output='softmax', input_dim=INPUT_DIM):
        if include_inputs:
            inputs = Input(shape=input_dim)
        x = Alexnet.getBase(inputs)
        x = Alexnet.getTop(x)
        x = Dense(class_num, activation=output)(x)
        model = Model(inputs, x)
        return model


batch_size = 32
NUM_CLASSES = 10
epochs = 100
data_augmentation = True
num_predictions = 20
INPUT_DIM = (32, 32, 3)

config = tf.ConfigProto( device_count = {'GPU': 1} ) 
sess = tf.Session(config=config) 
K.set_session(sess)

class Train():
    
    def initModel(self):
        self.model = Alexnet.set(
                    include_inputs=True,
                    class_num=NUM_CLASSES,
                    input_dim=INPUT_DIM,
                    output='sigmoid')
        self.model.summary()

    def train(self):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)
        x_train = x_train / 255 - 0.5
        x_test = x_test / 255 - 0.5
        optimizer = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        file = 'alexnet_cifar10-best.h5'
        if os.path.exists(file):
            self.model.load_weights(file)
        self.model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
        
        early_stop = EarlyStopping(monitor='val_loss', 
                           min_delta=0.001, 
                           patience=3, 
                           mode='min', 
                           verbose=1)

        checkpoint = ModelCheckpoint(file, 
                                    monitor='val_loss', 
                                    verbose=1, 
                                    save_best_only=True, 
                                    mode='min', 
                                    period=1)

        self.history = self.model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        callbacks        = [early_stop, checkpoint], 
                        validation_data=(x_test,y_test))
    
    def run(self):
        self.initModel()
        self.train()
      