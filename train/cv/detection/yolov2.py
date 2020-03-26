from model.cv.detection.yolov2 import Yolov2
from train.cv.classification.train import Train

from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation, Input
from tensorflow.keras.metrics import categorical_accuracy, top_k_categorical_accuracy, categorical_crossentropy
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

LABELS = ['RBC']

IMAGE_H, IMAGE_W = 416, 416
IMAGE_C          = 3
GRID_H,  GRID_W  = IMAGE_H / 32 , IMAGE_W / 32
BOX              = 5 # number of achor boxes
CLASS            = 20
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

wt_path = 'yolov2.weights'                      
train_image_folder = '/home/andy/data/coco/train2014/'
train_annot_folder = '/home/andy/data/coco/train2014ann/'
valid_image_folder = '/home/andy/data/coco/val2014/'
valid_annot_folder = '/home/andy/data/coco/val2014ann/'


if not os.path.exists(path_name): os.mkdir(path_name)

class Yolov2_Train(Train):

	def getData(self):
		pass

	def initModel(self):
		inputs = Input(shape=(IMAGE_H, IMAGE_W, IMAGE_C))
		true_boxes  = Input(shape=(1, 1, 1, TRUE_BOX_BUFFER , 4))
		self.model = Yolov2.set(inputs=[inputs, true_boxes],
								class_num=CLASS)

	def getCallback():
		early_stop = EarlyStopping(monitor='val_loss', 
		                           min_delta=0.001, 
		                           patience=3, 
		                           mode='min', 
		                           verbose=1)

		checkpoint = ModelCheckpoint('yolov2_voc.h5', 
		                             monitor='val_loss', 
		                             verbose=1, 
		                             save_best_only=True, 
		                             mode='min', 
		                             period=1)
		# Tensorboard for Keras
		## https://www.tensorflow.org/tensorboard/r2/scalars_and_keras
		logdir="logs/yolov2/" + datetime.now().strftime("%Y%m%d-%H%M%S")
		tensorboard = TensorBoard(log_dir=logdir, 
		                          histogram_freq=0, 
		                          write_graph=True, 
		                          write_images=False)
		callbacks = [early_stop, checkpoint, tensorboard]
		return callbacks

	def buildKerasGraph(self):
		optimizer = Adam(lr=0.5e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
		self.model.compile(loss=Yolov2_Train.custom_loss, optimizer=optimizer)

	def run(self):
		generator_config = {
		    'IMAGE_H'         : IMAGE_H, 
		    'IMAGE_W'         : IMAGE_W,
		    'GRID_H'          : GRID_H,  
		    'GRID_W'          : GRID_W,
		    'BOX'             : BOX,
		    'LABELS'          : LABELS,
		    'CLASS'           : len(LABELS),
		    'ANCHORS'         : ANCHORS,
		    'BATCH_SIZE'      : BATCH_SIZE,
		    'TRUE_BOX_BUFFER' : 50,
		    }
   		model.fit_generator(generator        = train_batch, 
                    steps_per_epoch  = len(train_batch), 
                    epochs           = 100, 
                    verbose          = 1,
                    validation_data  = valid_batch,
                    validation_steps = len(valid_batch),
                    callbacks        = Yolov2_Train.getCallback(), 
                    max_queue_size   = 3)

	def custom_loss(y_true, y_pred):
	    mask_shape = tf.shape(y_true)[:4]
	    
	    cell_x = tf.to_float(tf.reshape(tf.tile(tf.range(GRID_W), [GRID_H]), (1, GRID_H, GRID_W, 1, 1)))
	    cell_y = tf.transpose(cell_x, (0,2,1,3,4))

	    cell_grid = tf.tile(tf.concat([cell_x,cell_y], -1), [BATCH_SIZE, 1, 1, 5, 1])
	    
	    coord_mask = tf.zeros(mask_shape)
	    conf_mask  = tf.zeros(mask_shape)
	    class_mask = tf.zeros(mask_shape)
	    
	    seen = tf.Variable(0.)
	    total_recall = tf.Variable(0.)
	    
	    """
	    Adjust prediction
	    """
	    ### adjust x and y      
	    pred_box_xy = tf.sigmoid(y_pred[..., :2]) + cell_grid
	    
	    ### adjust w and h
	    pred_box_wh = tf.exp(y_pred[..., 2:4]) * np.reshape(ANCHORS, [1,1,1,BOX,2])
	    
	    ### adjust confidence
	    pred_box_conf = tf.sigmoid(y_pred[..., 4])
	    
	    ### adjust class probabilities
	    pred_box_class = y_pred[..., 5:]
	    
	    """
	    Adjust ground truth
	    """
	    ### adjust x and y
	    true_box_xy = y_true[..., 0:2] # relative position to the containing cell
	    
	    ### adjust w and h
	    true_box_wh = y_true[..., 2:4] # number of cells accross, horizontally and vertically
	    
	    ### adjust confidence
	    true_wh_half = true_box_wh / 2.
	    true_mins    = true_box_xy - true_wh_half
	    true_maxes   = true_box_xy + true_wh_half
	    
	    pred_wh_half = pred_box_wh / 2.
	    pred_mins    = pred_box_xy - pred_wh_half
	    pred_maxes   = pred_box_xy + pred_wh_half       
	    
	    intersect_mins  = tf.maximum(pred_mins,  true_mins)
	    intersect_maxes = tf.minimum(pred_maxes, true_maxes)
	    intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
	    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
	    
	    true_areas = true_box_wh[..., 0] * true_box_wh[..., 1]
	    pred_areas = pred_box_wh[..., 0] * pred_box_wh[..., 1]

	    union_areas = pred_areas + true_areas - intersect_areas
	    iou_scores  = tf.truediv(intersect_areas, union_areas)
	    
	    true_box_conf = iou_scores * y_true[..., 4]
	    
	    ### adjust class probabilities
	    true_box_class = tf.argmax(y_true[..., 5:], -1)

	    """
	    Determine the masks
	    """
	    ### coordinate mask: simply the position of the ground

	    coord_mask = tf.expand_dims(y_true[..., 4], axis=-1) * COORD_SCALE
	    
	    ### confidence mask: penelize predictors + penalize boxes with low IOU
	    # penalize the confidence of the boxes, which have IOU with some ground truth box < 0.6
	    true_xy = true_boxes[..., 0:2]
	    true_wh = true_boxes[..., 2:4]
	    
	    true_wh_half = true_wh / 2.
	    true_mins    = true_xy - true_wh_half
	    true_maxes   = true_xy + true_wh_half
	    
	    pred_xy = tf.expand_dims(pred_box_xy, 4)
	    pred_wh = tf.expand_dims(pred_box_wh, 4)
	    
	    pred_wh_half = pred_wh / 2.
	    pred_mins    = pred_xy - pred_wh_half
	    pred_maxes   = pred_xy + pred_wh_half    
	    
	    intersect_mins  = tf.maximum(pred_mins,  true_mins)
	    intersect_maxes = tf.minimum(pred_maxes, true_maxes)
	    intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
	    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
	    
	    true_areas = true_wh[..., 0] * true_wh[..., 1]
	    pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

	    union_areas = pred_areas + true_areas - intersect_areas
	    iou_scores  = tf.truediv(intersect_areas, union_areas)

	    best_ious = tf.reduce_max(iou_scores, axis=4)
	    conf_mask = conf_mask + tf.to_float(best_ious < 0.6) * (1 - y_true[..., 4]) * NO_OBJECT_SCALE
	    
	    # penalize the confidence of the boxes, which are reponsible for corresponding ground truth box
	    conf_mask = conf_mask + y_true[..., 4] * OBJECT_SCALE
	    
	    ### class mask: simply the position of the ground truth boxes (the predictors)
	    class_mask = y_true[..., 4] * tf.gather(CLASS_WEIGHTS, true_box_class) * CLASS_SCALE       
	    
	    """
	    Warm-up training
	    """
	    no_boxes_mask = tf.to_float(coord_mask < COORD_SCALE/2.)

	    seen = tf.assign_add(seen, 1.)
	    
	    true_box_xy, true_box_wh, coord_mask = tf.cond(tf.less(seen, WARM_UP_BATCHES), 
	                          lambda: [true_box_xy + (0.5 + cell_grid) * no_boxes_mask, 
	                                   true_box_wh + tf.ones_like(true_box_wh) * np.reshape(ANCHORS, [1,1,1,BOX,2]) * no_boxes_mask, 
	                                   tf.ones_like(coord_mask)],
	                          lambda: [true_box_xy, 
	                                   true_box_wh,
	                                   coord_mask])
	    
	    """
	    Finalize the loss
	    """
	    nb_coord_box = tf.reduce_sum(tf.to_float(coord_mask > 0.0))
	    nb_conf_box  = tf.reduce_sum(tf.to_float(conf_mask  > 0.0))
	    nb_class_box = tf.reduce_sum(tf.to_float(class_mask > 0.0))
	    
	    loss_xy    = tf.reduce_sum(tf.square(true_box_xy-pred_box_xy)     * coord_mask) / (nb_coord_box + 1e-6) / 2.
	    loss_wh    = tf.reduce_sum(tf.square(true_box_wh-pred_box_wh)     * coord_mask) / (nb_coord_box + 1e-6) / 2.
	    loss_conf  = tf.reduce_sum(tf.square(true_box_conf-pred_box_conf) * conf_mask)  / (nb_conf_box  + 1e-6) / 2.
	    loss_class = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_box_class, logits=pred_box_class)
	    loss_class = tf.reduce_sum(loss_class * class_mask) / (nb_class_box + 1e-6)
	    
	    loss = loss_xy + loss_wh + loss_conf + loss_class
	    
	    nb_true_box = tf.reduce_sum(y_true[..., 4])
	    nb_pred_box = tf.reduce_sum(tf.to_float(true_box_conf > 0.5) * tf.to_float(pred_box_conf > 0.3))

	    """
	    Debugging code
	    """    
	    current_recall = nb_pred_box/(nb_true_box + 1e-6)
	    total_recall = tf.assign_add(total_recall, current_recall) 

	    loss = tf.Print(loss, [tf.zeros((1))], message='Dummy Line \t', summarize=1000)
	    loss = tf.Print(loss, [loss_xy], message='Loss XY \t', summarize=1000)
	    loss = tf.Print(loss, [loss_wh], message='Loss WH \t', summarize=1000)
	    loss = tf.Print(loss, [loss_conf], message='Loss Conf \t', summarize=1000)
	    loss = tf.Print(loss, [loss_class], message='Loss Class \t', summarize=1000)
	    loss = tf.Print(loss, [loss], message='Total Loss \t', summarize=1000)
	    loss = tf.Print(loss, [current_recall], message='Current Recall \t', summarize=1000)
	    loss = tf.Print(loss, [total_recall/seen], message='Average Recall \t', summarize=1000)
    
    return loss