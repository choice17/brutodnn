from model.cv.classification.mobilenetv2 import Mobilenetv2
from train.cv.classification.train import Train
from data.voc import VOC

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint

from utils.utils import imshow
import tensorflow as tf
import numpy as np
#from sklearn.metrics import confusion_matrix

import os
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices)      

# https://datascience.stackexchange.com/questions/13490/how-to-set-class-weights-for-imbalanced-classes-in-keras
"""
conf_matr = np.zeros((NUM_CLASSES,2,2))
rate = 0.5
for x_batch, y_batch in valid_batch:
    pred = t.model.predict(x_batch)
    pred[pred >= rate] = 1
    pred[pred < rate] = 0
    for i in range(NUM_CLASSES):
      cm = confusion_matrix(pred[:,i:i+1], y_batch[:,i:i+1],[1,0])
      conf_matr[i] += np.array(cm)

"""
"""
https://github.com/keras-team/keras/issues/2115
def w_categorical_crossentropy(y_true, y_pred, weights):
    nb_cl = len(weights)
    final_mask = K.zeros_like(y_pred[:, 0])
    y_pred_max = K.max(y_pred, axis=1)
    y_pred_max = K.reshape(y_pred_max, (K.shape(y_pred)[0], 1))
    y_pred_max_mat = K.cast(K.equal(y_pred, y_pred_max), K.floatx())
    for c_p, c_t in product(range(nb_cl), range(nb_cl)):
        final_mask += (weights[c_t, c_p] * y_pred_max_mat[:, c_p] * y_true[:, c_t])
    return K.categorical_crossentropy(y_pred, y_true) * final_mask

w_array = np.ones((10,10))
w_array[1, 7] = 1.2
w_array[7, 1] = 1.2

ncce = partial(w_categorical_crossentropy, weights=np.ones((10,10)))
compile(loss = ncce)
"""

BATCH_SIZE = 8
NUM_CLASSES = 20
EPOCHS = 1
DATASET = "."
INPUT_DIM = (224, 224, 3)

config = {
    'dataset': 'voc',
    'data_mode': 'none',
    'model_info': 'pretained',
    'batch_size': BATCH_SIZE,
    'img_h': INPUT_DIM[0],
    'img_w': INPUT_DIM[1],
    'img_c': INPUT_DIM[2],
    'num_class': NUM_CLASSES,
    'epochs': EPOCHS,
    'shuffle': 1,
    'do_augment':1
}

path_name = "tmp"
model_name = "mobilenet"
file_name = "mobilenetv2.pb"
log_path = 'logs'

def K_binary_loss(y_true, y_pred):
    loss = K.binary_crossentropy(y_true, y_pred)
    #loss2 = K.categorical_crossentropy(t, p)
    loss = K.cast(K.equal(y_true,1), K.floatx()) * loss + loss
    return loss

if not os.path.exists(path_name): os.mkdir(path_name)
if not os.path.exists(log_path): os.mkdir(log_path)

class Mobilenet_Train(Train):

    def __init__(self):
        self.config = config

    def getData(self):
        if self.config['dataset'] == 'voc':
            self.data_model = VOC()
            self.data_model.getFileList()
            self.data_model.getAnnot()
            self.data_model.getClassAnnot()
            self.data_model.setGeneratorConfig(self.config)

    def initModel(self):
        _config = tf.ConfigProto( device_count = {'GPU': 1} ) 
        sess = tf.Session(config=_config) 
        K.set_session(sess)
        if self.config['model_info'] == 'begin':
            self.model = Mobilenetv2.set(
                            include_inputs=True,
                            class_num=NUM_CLASSES,
                            input_dim=INPUT_DIM,
                            output='sigmoid'
                            )
        elif self.config['model_info'] == 'pretained':
            self.model = Mobilenetv2.getKerasModelBase(
                    num_class=NUM_CLASSES,
                    output='sigmoid',
                    fix_layer=156
                    )
        self.model.summary()

    def buildTrainKeras(self):
        optimizer = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        #optimizer = SGD(lr=1e-4, decay=0.0005, momentum=0.9)
        #optimizer = RMSprop(lr=1e-5, rho=0.9, epsilon=1e-08, decay=0.0)

        self.model.compile(loss=K_binary_loss,
              optimizer=optimizer,
              metrics=['accuracy'])
        """
        self.model.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
              """

    def fit(self):
        file = path_name + '/' + model_name + '-best.h5'
        if os.path.exists(file):
            self.model.load_weights(file)
            fix_layer = 156
            for layer in self.model.layers:
                layer.trainable=False
            # or if we want to set the first 20 layers of the network to be non-trainable
            for layer in self.model.layers[:fix_layer]:
                layer.trainable=False
            for layer in self.model.layers[fix_layer:]:
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
        tensorboard = TensorBoard(log_dir=log_path +'/' + model_name + '_' + str(tb_counter), 
                                histogram_freq=0, 
                                write_graph=True, 
                                write_images=False)

        train_batch = self.data_model.getTrainBatch()
        valid_batch = self.data_model.getValidBatch()

        self.model.fit_generator(
                    generator        = train_batch, 
                    steps_per_epoch  = len(train_batch), 
                    epochs           = self.config['epochs'], 
                    verbose          = 1,
                    validation_data  = valid_batch,
                    validation_steps = len(valid_batch),
                    callbacks        = [early_stop, checkpoint, tensorboard], 
                    max_queue_size   = 3)
        self.valid_batch = valid_batch

    def run(self):
        self.initModel()
        self.getData()
        self.buildTrainKeras()
        self.fit()
        self.save(file_name=file_name, path_name=path_name)

    """
    def evaluate(self):
        conf_matr = np.zeros((NUM_CLASSES,NUM_CLASSES))
        for x_batch, y_batch in self.valid_batch:
            pred = self.model.predict(x_batch)
            cm = confusion_matrix(pred, y_batch)
            conf_matr += np.array(cm)
        print(conf_matr)
    """