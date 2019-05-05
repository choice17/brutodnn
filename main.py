#from train.cv.classification.lenet import main
#from train.cv.classification.alexnet import Alexnet_Train
from train.cv.classification.mobilenet import Mobilenet_Train
#main()
train = Mobilenet_Train()
train.run()
print()
