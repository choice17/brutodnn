#from train.cv.classification.lenet import main
from train.cv.classification.alexnet import Alexnet_Train
from train.cv.classification.mobilenet import Mobilenet_Train
import sys
"""
if len(sys.argv) == 2:
    if sys.argv[1] == 'mobilenet':
        print("selected mobilenet")
        train = Mobilenet_Train()
    elif sys.argv[1] == 'alexnet':
        print("Selected Alexnet")
        train = Alexnet_Train()
else:
#main()
    train = Mobilenet_Train()
"""
train = Alexnet_Train()
train.run()
print()
