from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation, Input
#from tensorflow.keras.metrics import categorical_accuracy, top_k_categorical_accuracy, categorical_crossentropy
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.callbacks import EarlyStopping, Tensorboard, ModelCheckpoint
#from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Model
from model.base.cv.yolov2 import Yolov2_Base
from model.top.detection.yolov2 import Yolov2_Top

from utils.utils import _sigmoid, _softmax, weight_reader

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

LABELS = ['RBC']

IMAGE_H, IMAGE_W = 416, 416
IMAGE_C          = 3
GRID_H,  GRID_W  = IMAGE_H / 32 , IMAGE_W / 32
BOX              = 5 # number of achor boxes
CLASS            = len(LABELS)
CLASS_WEIGHTS    = np.ones(CLASS, dtype='float32')
OBJ_THRESHOLD    = 0.3
NMS_THRESHOLD    = 0.3
ANCHORS          = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]

#NO_OBJECT_SCALE  = 1.0
#OBJECT_SCALE     = 5.0
#COORD_SCALE      = 1.0
#CLASS_SCALE      = 1.0

#BATCH_SIZE       = 16
WARM_UP_BATCHES  = 100
TRUE_BOX_BUFFER  = 50


class Yolov2(object):

	def getBase(inputs):
		base, skip_connection = Yolov2_Base.set(inputs)
		return base, skip_connection

	def getTop(inputs):
		return Yolov2_Top.set(inputs=inputs)

	def set(inputs=None, class_num=CLASS, include_input=False, train=True):
		if include_input:
			inputs = Input(shape=(IMAGE_H, IMAGE_W, IMAGE_C))
			true_boxes  = Input(shape=(1, 1, 1, TRUE_BOX_BUFFER , 4))
		base, skip_connection, idx = Yolov2.getBase(inputs)
		top, idx                   = Yolov2.getTop(inputs=[base, skip_connection], idx=idx+1)
		conv_n = 'conv_%d' % (idx + 1)
		# Layer 23 (Dim:13x13x1024, /32)
		x = Conv2D(BOX * (4 + 1 + CLASS), (1,1), strides=(1,1), padding='same', name=conv_n)(x)
		predict = Reshape((GRID_H, GRID_W, BOX, 4 + 1 + CLASS),name='predict')(x)

		if train:
			# small hack to allow true_boxes to be registered when Keras build the model 
			# for more information: https://github.com/fchollet/keras/issues/2790
			output = Lambda(lambda args: args[0])([output, true_boxes])
			model = Model([inputs, true_boxes], output)
		else:
			model = Model(inputs, predict)
		self.idx = idx
		return model

	def decode_netout(netout, anchors, nb_class, obj_threshold=0.3, nms_threshold=0.3):
		grid_h, grid_w, nb_box = netout.shape[:3]

		boxes = []

		# decode the output by the network
		netout[..., 4]  = _sigmoid(netout[..., 4])
		netout[..., 5:] = netout[..., 4][..., np.newaxis] * _softmax(netout[..., 5:])
		netout[..., 5:] *= netout[..., 5:] > obj_threshold

		for row in range(grid_h):
		    for col in range(grid_w):
		        for b in range(nb_box):
		            # from 4th element onwards are confidence and class classes
		            classes = netout[row,col,b,5:]
		            
		            if np.sum(classes) > 0:
		                # first 4 elements are x, y, w, and h
		                x, y, w, h = netout[row,col,b,:4]

		                x = (col + _sigmoid(x)) / grid_w # center position, unit: image width
		                y = (row + _sigmoid(y)) / grid_h # center position, unit: image height
		                w = anchors[2 * b + 0] * np.exp(w) / grid_w # unit: image width
		                h = anchors[2 * b + 1] * np.exp(h) / grid_h # unit: image height
		                confidence = netout[row,col,b,4]
		                
		                box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, confidence, classes)
		                
		                boxes.append(box)

		# suppress non-maximal boxes
		for c in range(nb_class):
		    sorted_indices = list(reversed(np.argsort([box.classes[c] for box in boxes])))

		    for i in range(len(sorted_indices)):
		        index_i = sorted_indices[i]
		        
		        if boxes[index_i].classes[c] == 0: 
		            continue
		        else:
		            for j in range(i+1, len(sorted_indices)):
		                index_j = sorted_indices[j]
		                
		                if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_threshold:
		                    boxes[index_j].classes[c] = 0
		                    
		# remove the boxes which are less likely than a obj_threshold
		boxes = [box for box in boxes if box.get_score() > obj_threshold]

		return boxes    

	def load_weight(path, idx=23):
		weight_reader = WeightReader(wt_path)
		weight_reader.reset()
		nb_conv = idx
		for i in range(1, nb_conv+1):
		    conv_layer = model.get_layer('conv_' + str(i))
		    
		    if i < nb_conv:
		        norm_layer = model.get_layer('norm_' + str(i))
		        
		        size = np.prod(norm_layer.get_weights()[0].shape)

		        beta  = weight_reader.read_bytes(size)
		        gamma = weight_reader.read_bytes(size)
		        mean  = weight_reader.read_bytes(size)
		        var   = weight_reader.read_bytes(size)

		        weights = norm_layer.set_weights([gamma, beta, mean, var])       
		        
		    if len(conv_layer.get_weights()) > 1:
		        bias   = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[1].shape))
		        kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
		        kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
		        kernel = kernel.transpose([2,3,1,0])
		        conv_layer.set_weights([kernel, bias])
		    else:
		        kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
		        kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
		        kernel = kernel.transpose([2,3,1,0])
		        conv_layer.set_weights([kernel])

