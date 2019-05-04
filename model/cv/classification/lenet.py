# tensorflow v1.10
#from tensorflow import keras
#from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation, Input
#from tensorflow.keras.metrics import categorical_accuracy, top_k_categorical_accuracy, categorical_crossentropy
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
#from tensorflow.keras.optimizers import Adam
#from tensorflow.keras.applications import MobileNet

from tensorflow.keras.models import Model
from model.base.cv.lenet import Lenet_Base
from model.top.classification.lenet import Lenet_Top

INPUT_DIM = (28, 28, 1)
CLASS_NO = 10

class Lenet(object):

    def set(inputs=None, include_input=False, class_num=CLASS_NO, output="softmax"):
        """
        input should be 32x32x1 input
        """
        assert output in ['softmax', 'sigmoid'], "not valid output format"
        if include_input:
            inputs = Input(shape=INPUT_DIM)
        base = Lenet_Base.set(inputs)
        top = Lenet_Top.set(base)
        output = Dense(units=class_num, activation=output)(top)
        lenet = Model(inputs=inputs, outputs=output)
        return lenet

    def loadKerasModel(self):
        pass
    
    def loadPbModel(self):
        pass
