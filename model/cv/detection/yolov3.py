from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation, Input
#from tensorflow.keras.metrics import categorical_accuracy, top_k_categorical_accuracy, categorical_crossentropy
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.callbacks import EarlyStopping, Tensorboard, ModelCheckpoint
#from tensorflow.keras.optimizers import Adam
import tensorflow as tf

from tensorflow.keras.models import Model
from model.base.cv.yolov3 import Yolov3_Base
from model.top.detection.yolov3 import Yolov3_Top
from utils.utils import WeightReaderYolov3

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

NO_OBJECT_SCALE  = 1.0
OBJECT_SCALE     = 5.0
COORD_SCALE      = 1.0
CLASS_SCALE      = 1.0

BATCH_SIZE       = 16
WARM_UP_BATCHES  = 100
TRUE_BOX_BUFFER  = 50


class Yolov3(object):

	def _conv_block(inp, convs, skip=True):
	    x = inp
	    count = 0

	    for conv in convs:
	        if count == (len(convs) - 2) and skip:
	            skip_connection = x
	        count += 1
	        
	        if conv['kernel'] > 1: x = ZeroPadding2D(1)(x)
	        x = Conv2D(conv['filter'], 
	                   conv['kernel'], 
	                   strides=conv['stride'], 
	                   padding='valid', 
	                   name='conv_' + str(conv['layer_idx']), 
	                   use_bias=False if conv['bnorm'] else True,
	                   kernel_regularizer=tf.keras.regularizers.l2(0.001) if conv['l2'] else None)(x)
	        if conv['bnorm']: x = BatchNormalization(epsilon=0.001, name='bnorm_' + str(conv['layer_idx']))(x)
	        if conv['leaky']: x = LeakyReLU(alpha=0.1, name='leaky_' + str(conv['layer_idx']))(x)

	    return add([skip_connection, x]) if skip else x

	def getBase(inputs, conv_block):
		base, skip_connection = Yolov3_Base.set(inputs, _conv_block=conv_block)
		return base, skip_connection

	def getTop(inputs, conv_block):
		return Yolov2_Top.set(inputs=inputs, _conv_block=conv_block)

	def set(inputs=None, class_num=CLASS, include_input=False, train=True):
		if include_input:
			inputs = Input(shape=(IMAGE_H, IMAGE_W, IMAGE_C))
			true_boxes  = Input(shape=(1, 1, 1, TRUE_BOX_BUFFER , 4))
		bases = Yolov3.getBase(inputs, conv_block=Yolov3._conv_block)
		top = Yolov3.getTop(inputs=bases, conv_block=Yolov3._conv_block)

		model = Model(inputs=inputs, outputs=top)
		return model

	def preprocess_input(image, net_h, net_w):
	    new_h, new_w, _ = image.shape

	    # determine the new size of the image
	    if (float(net_w)/new_w) < (float(net_h)/new_h):
	        new_h = (new_h * net_w)/new_w
	        new_w = net_w
	    else:
	        new_w = (new_w * net_h)/new_h
	        new_h = net_h

	    # resize the image to the new size
	    resized = cv2.resize(image[:,:,::-1]/255., (new_w, new_h))

	    # embed the image into the standard letter box
	    new_image = np.ones((net_h, net_w, 3)) * 0.5
	    new_image[(net_h-new_h)/2:(net_h+new_h)/2, (net_w-new_w)/2:(net_w+new_w)/2, :] = resized
	    new_image = np.expand_dims(new_image, 0)

	    return new_image

	def decode_netout(netout, anchors, obj_thresh, nms_thresh, net_h, net_w):
	    grid_h, grid_w = netout.shape[:2]
	    nb_box = 3
	    netout = netout.reshape((grid_h, grid_w, nb_box, -1))
	    nb_class = netout.shape[-1] - 5

	    boxes = []

	    netout[..., :2]  = _sigmoid(netout[..., :2])
	    netout[..., 4:]  = _sigmoid(netout[..., 4:])
	    netout[..., 5:]  = netout[..., 4][..., np.newaxis] * netout[..., 5:]
	    netout[..., 5:] *= netout[..., 5:] > obj_thresh

	    for i in range(grid_h*grid_w):
	        row = i / grid_w
	        col = i % grid_w
	        
	        for b in range(nb_box):
	            # 4th element is objectness score
	            objectness = netout[row, col, b, 4]
	            
	            if(objectness <= obj_thresh): continue
	            
	            # first 4 elements are x, y, w, and h
	            x, y, w, h = netout[row,col,b,:4]

	            x = (col + x) / grid_w # center position, unit: image width
	            y = (row + y) / grid_h # center position, unit: image height
	            w = anchors[2 * b + 0] * np.exp(w) / net_w # unit: image width
	            h = anchors[2 * b + 1] * np.exp(h) / net_h # unit: image height  
	            
	            # last elements are class probabilities
	            classes = netout[row,col,b,5:]
	            
	            box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, objectness, classes)

	            boxes.append(box)

	    return boxes

	def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):
	    if (float(net_w)/image_w) < (float(net_h)/image_h):
	        new_w = net_w
	        new_h = (image_h*net_w)/image_w
	    else:
	        new_h = net_w
	        new_w = (image_w*net_h)/image_h
	        
	    for i in range(len(boxes)):
	        x_offset, x_scale = (net_w - new_w)/2./net_w, float(new_w)/net_w
	        y_offset, y_scale = (net_h - new_h)/2./net_h, float(new_h)/net_h
	        
	        boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
	        boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
	        boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
	        boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)

	def do_nms(boxes, nms_thresh):
	    if len(boxes) > 0:
	        nb_class = len(boxes[0].classes)
	    else:
	        return
	        
	    for c in range(nb_class):
	        sorted_indices = np.argsort([-box.classes[c] for box in boxes])

	        for i in range(len(sorted_indices)):
	            index_i = sorted_indices[i]

	            if boxes[index_i].classes[c] == 0: continue

	            for j in range(i+1, len(sorted_indices)):
	                index_j = sorted_indices[j]

	                if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
	                    boxes[index_j].classes[c] = 0

	def draw_boxes(image, boxes, labels, obj_thresh):
	    for box in boxes:
	        label_str = ''
	        label = -1
	        
	        for i in range(len(labels)):
	            if box.classes[i] > obj_thresh:
	                label_str += labels[i]
	                label = i
	                print(labels[i] + ': ' + str(box.classes[i]*100) + '%')
	                
	        if label >= 0:
	            cv2.rectangle(image, (box.xmin,box.ymin), (box.xmax,box.ymax), (0,255,0), 3)
	            cv2.putText(image, 
	                        label_str + ' ' + str(box.get_score()), 
	                        (box.xmin, box.ymin - 13), 
	                        cv2.FONT_HERSHEY_SIMPLEX, 
	                        1e-3 * image.shape[0], 
	                        (0,255,0), 2)
	        
	    return image      

	def load_weight(path, yolov3model):
		weight_reader = WeightReader(path)
	    weight_reader.load_weights(yolov3model)