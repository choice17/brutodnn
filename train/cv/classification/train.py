from tensorflow.keras.optimizers import Adam

class Train(object):
    def __init__(self, model):
        self.model = model

    def buildTraining(self):
    	pass

    def buildTrainKeras(self):
        self.model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

    def buildTrainTf(self):
    	pass

    def fit(self):
    	pass

    def eval(self):
    	pass



