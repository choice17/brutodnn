from model.cv.classification.mobilenetv2 import Mobilenetv2
from train.cv.classification.train import Train
from data.voc import VOC

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint


import tensorflow as tf
import numpy as np

import os
from tensorflow.python.client import device_lib

BATCH_SIZE = 4
NUM_CLASSES = 20
EPOCHS = 2
DATASET = "."
INPUT_DIM = (224, 224, 3)

config = {
    'batch_size': BATCH_SIZE,
    'img_h': INPUT_DIM[0],
    'img_w': INPUT_DIM[1],
    'img_c': INPUT_DIM[2],
    'num_class': NUM_CLASSES,
    'epochs': EPOCHS,
    'shuffle': 1
}

class Mobilenet_Train(Train):

    def __init__(self):
        self.config = config

    def getData(self):
        self.voc = VOC()
        self.voc.getFileList()
        self.voc.getAnnot()
        self.voc.getClassAnnot()
        self.voc.setGeneratorConfig(self.config)

    def initModel(self):
        """self.model = Mobilenetv2.set(
                        include_inputs=True,
                        class_num=NUM_CLASSES,
                        input_dim=INPUT_DIM,
                        output='sigmoid'
                        )"""
        self.model = Mobilenetv2.getKerasModelBase(
                num_class=NUM_CLASSES,
                output='sigmoid',
                fix_layer=20
                )
        self.model.summary()

    def buildTrainKeras(self):
        optimizer = Adam(lr=5e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        #optimizer = SGD(lr=1e-4, decay=0.0005, momentum=0.9)
        #optimizer = RMSprop(lr=1e-5, rho=0.9, epsilon=1e-08, decay=0.0)
        self.model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

    def fit(self):
        file = 'tmp/mobilenetv2-best.h5'
        if os.path.exists(file):
            self.model.load(file)
            fix_layer = 20
            for layer in model.layers:
                layer.trainable=False
            # or if we want to set the first 20 layers of the network to be non-trainable
            for layer in model.layers[:fix_layer]:
                layer.trainable=False
            for layer in model.layers[fix_layer:]:
                layer.trainable=True

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
        tb_counter  = 1
        tensorboard = TensorBoard(log_dir='logs/' + 'mobilenetv2' + '_' + str(tb_counter), 
                                histogram_freq=0, 
                                write_graph=True, 
                                write_images=False)

        train_batch = self.voc.getTrainBatch()
        valid_batch = self.voc.getValidBatch()

        self.model.fit_generator(
                    generator        = train_batch, 
                    steps_per_epoch  = len(train_batch), 
                    epochs           = self.config['epochs'], 
                    verbose          = 1,
                    validation_data  = valid_batch,
                    validation_steps = len(valid_batch),
                    callbacks        = [early_stop, checkpoint, tensorboard], 
                    max_queue_size   = 3)

    def run(self):
        self.initModel()
        self.getData()
        self.buildTrainKeras()
        self.fit()
        self.save()