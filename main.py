#from train.cv.classification.lenet import main
from train.cv.classification.alexnet import Alexnet_Train
from train.cv.classification.mobilenet import Mobilenet_Train
import sys

if len(sys.argv) == 3:
    if sys.argv[2] == 'mobilenet':
        train = Mobilenet_Train()
    elif sys.argv[2] == 'alexnet':
        train = Alexnet_Train()
else:
#main()
    train = Mobilenet_Train()
train.run()
print()
