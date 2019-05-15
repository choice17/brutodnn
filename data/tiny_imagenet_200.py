import numpy as np
import xml.etree.ElementTree as ET
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from utils.utils import imshow
import os

data_path = 'data/dataset/tiny-imagenet-200/'
class_path = 'wnids.txt'
train_path = 'train/'
val_path = 'val/'
val_annot_path = 'val_annotations.txt'
test_path = 'test/'
images_path = 'images/'

annot_file_fmt = '{}/{}_boxes.txt'
val_img_list = []
test_img_list = []
train_img_list = []
train_label_list = []
val_label_list = []
class_num = 200

BATCH_SIZE = 4
NUM_CLASSES = class_num
EPOCHS = 1
DATASET = "."
INPUT_DIM = (64, 64, 3)

config = {
    'dataset': 'voc',
    'data_mode': 'none',
    'batch_size': BATCH_SIZE,
    'img_h': INPUT_DIM[0],
    'img_w': INPUT_DIM[1],
    'img_c': INPUT_DIM[2],
    'num_class': NUM_CLASSES,
    'epochs': EPOCHS,
    'shuffle': 1,
    'do_augment': 0
}

class TINY_IMAGENET200(object):
    class_num = class_num
    class_path = class_path
    train_path = train_path
    val_path = val_path
    val_annot_path = val_annot_path
    test_path = test_path
    images_path = images_path
    datagen = ImageDataGenerator(
                featurewise_center=False,
                samplewise_center=False,
                featurewise_std_normalization=False,
                samplewise_std_normalization=False,
                zca_whitening=False,
                zca_epsilon=1e-6,
                #rotation_range=20,
                #width_shift_range=0.1,
                #height_shift_range=0.1,
                #shear_range=0.3,
                #zoom_range=0.1,
                channel_shift_range=30,
                fill_mode='nearest',
                cval=0.,
                horizontal_flip=True,
                vertical_flip=False,
                rescale=None,
                preprocessing_function=None)

    def __init__(self):
        self.dirs = data_path

    def set(self, data_path):
        self.dirs = data_path

    def getFileList(self):
        train_img_list = []
        train_annot_list = []
        with open(data_path + class_path, 'r') as f:
            self.class_list = f.read().split('\n')
        for _cls in self.class_list:
            if _cls:
                cls_path = '{}{}{}/'.format(self.dirs, self.train_path, _cls)
                cls_list = os.listdir(cls_path)
                #for img_path in cls_list:
                #    img_path = '%s%s'.format(cls_path, img_path)
                #    train_img_list.append(img_path)
                label_path = annot_file_fmt.format(cls_path, _cls)
                with open(label_path, 'r') as f:
                    annot_path = f.read().split('\n')
                for entry in annot_path:
                    if entry == []:
                        continue
                    if entry == ['']:
                        continue
                    annot = entry.split('\t')
                    img_path = '{}{}{}{}'.format(self.dirs,self.train_path,self.images_path, annot[0])
                    train_img_list.append(img_path)
                    cls_info = [annot[0].split('_')[0]]
                    train_annot_list.append(cls_info + annot[1:])

        self.train_img_list = train_img_list
        self.train_annot_list = train_annot_list

        val_img_list = []
        val_annot_list = []
        #val_img_path = '%s%s%s'.format(self.dirs, self.val_path, self.images_path)
        #val_img_list = os.listdir(val_img_path)
        val_annot_path = '{}{}{}'.format(self.dirs, self.val_path, self.val_annot_path)
        with open(val_annot_path, 'r') as f:
            annot_list = f.read().split('\n')
        for entry in annot_list:
            if entry == []:
                continue
            if entry == ['']:
                continue
            annot = entry.split('\t')
            img_path = '{}{}{}{}'.format(self.dirs,self.val_path, self.images_path, annot[0])
            val_img_list.append(img_path)
            val_annot_list.append(annot[1:])
        self.val_img_list = val_img_list
        self.val_annot_list = val_annot_list
    
        self.class_key = {key:item for item, key in enumerate(self.class_list)}

    def getClassAnnot(self):
        _len = len(self.train_img_list)
        self.train_label = np.zeros((_len, TINY_IMAGENET200.class_num))
        i = 0
        for annot in self.train_annot_list:
            self.train_label[i, self.class_key[annot[0]]] = 1
            i += 1
        _len = len(self.val_img_list)
        i = 0
        self.val_label = np.zeros((_len, TINY_IMAGENET200.class_num))
        for annot in self.val_annot_list:
            self.val_label[i, self.class_key[annot[0]]] = 1
            i += 1

    def setGeneratorConfig(self, config=config):
        """
        batch_size : 64
        img_h : 64
        img_w : 64
        img_c : 3
        norm : 1
        num_classes: 200
        shuffle : 0
        """
        self.config = config
        self.shuffle = config['shuffle']
        if self.config['data_mode'] == 'on_memory':
            self.getImage()

    def getImage(self):
        print("[INFO] Load image from list to memory ...")
        _len = len(self.train_annot)
        self.train_images = np.zeros((_len,
                            self.config['img_h'],
                            self.config['img_w'],
                            self.config['img_c']))
        for j in range(_len):
            img_path = self.train_img_list[j]
            img = cv2.imread(img_path)[:,:,::-1]
            img = cv2.resize(img, (self.config['img_w'],self.config['img_h']))
            self.train_images[j, ...] = img
        
        _len = len(self.valid_annot)
        self.valid_images = np.zeros((_len,
                            self.config['img_h'],
                            self.config['img_w'],
                            self.config['img_c']))
        for j in range(_len):
            img_path = self.train_img_list[j]
            img = cv2.imread(img_path)
            img = cv2.resize(img, (self.config['img_w'],self.config['img_h']))
            self.valid_images[j, ...] = img

    def getTrainBatch(self):
        if self.config['data_mode'] == 'on_memory':
            return self.train_image, self.train_label
        return TRAIN_BATCH(self)

    def getValidBatch(self):
        if self.config['data_mode'] == 'on_memory':
            return self.valid_images, self.valid_label
        return VAL_BATCH(self)

    def preprocessData(img):
        #img = img.astype(np.float32) / 255 - 0.5
        return img
    
    def augment_data(img_batch, y_batch, batch_size):
        img_batch, y_batch = TINY_IMAGENET200.datagen.flow((img_batch, y_batch), batch_size=batch_size).next()
        return img_batch, y_batch

class TRAIN_BATCH(Sequence):
    def __init__(self, tiny_imagenet200):
        self.config = tiny_imagenet200.config
        self.train_label = tiny_imagenet200.train_label
        self.train_img_list = tiny_imagenet200.train_img_list
        self.shuffle = tiny_imagenet200.config['shuffle']
        self.generate_list = list(range(len(self)))
        self.augment_data = 0
        print("[INFO] total %d training images" % len(tiny_imagenet200.train_img_list))

    def __len__(self):
        return int((self.train_label.shape[0] / self.config['batch_size']) + 0.5)

    def size(self):
        return self.train_label.shape[0]

    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.generate_list)
        if self.config['do_augment']: self.augment_data = 1

    def __getitem__(self, idx):
        x_batch = np.empty((self.config['batch_size'],
                            self.config['img_h'],
                            self.config['img_w'],
                            self.config['img_c']))
        y_batch = np.empty((self.config['batch_size'],
                            self.config['num_class']))
        idx = self.generate_list[idx]
        idx_base = idx * self.config['batch_size']
        idx_top = idx_base + self.config['batch_size']
        if idx_top > self.size():
            idx_top = self.size()
            size = idx_top - idx_base
            x_batch = x_batch[:size, ...]
            y_batch = y_batch[:size, ...]
        i = 0
        for j in range(idx_base, idx_top):
            img_path = self.train_img_list[j]
            img = cv2.imread(img_path)[:,:,::-1]
            img = cv2.resize(img, (self.config['img_w'], self.config['img_h']))
            x_batch[i, ...] = img
            y_batch[i, ...] = self.train_label[j, ...]
            i += 1
        if self.augment_data:
            x_batch, y_batch = TINY_IMAGENET200.augment_data(x_batch, y_batch, batch_size=self.config['batch_size'])
        x_batch = TINY_IMAGENET200.preprocessData(x_batch)
        return x_batch, y_batch

class VAL_BATCH(Sequence):
    def __init__(self, tiny_imagenet200):
        self.config = tiny_imagenet200.config
        self.valid_label = tiny_imagenet200.valid_label
        self.valid_img_list = tiny_imagenet200.valid_img_list
        self.shuffle = tiny_imagenet200.config['shuffle']
        self.generate_list = list(range(len(self)))
        print("[INFO] total %d validation images" % len(tiny_imagenet200.valid_img_list))

    def __len__(self):
        return int((self.valid_label.shape[0] / self.config['batch_size']) + 0.5)

    def size(self):
        return self.valid_label.shape[0]

    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.generate_list)
        self.augment_data = 1

    def __getitem__(self, idx):
        x_batch = np.empty((self.config['batch_size'],
                            self.config['img_h'],
                            self.config['img_w'],
                            self.config['img_c']))
        y_batch = np.empty((self.config['batch_size'],
                            self.config['num_class']))

        #for idx in self.generate_list:
        idx_base = idx * self.config['batch_size']
        idx_top = idx_base + self.config['batch_size']
        if idx_top > self.size():
            idx_top = self.size()
            size = idx_top - idx_base
            x_batch = x_batch[:size, ...]
            y_batch = y_batch[:size, ...]
        i = 0
        for j in range(idx_base, idx_top):
            img_path = self.valid_img_list[j]
            img = cv2.imread(img_path)[:,:,::-1]
            img = cv2.resize(img, (self.config['img_w'], self.config['img_h']))
            x_batch[i, ...] = img
            y_batch[i, ...] = self.valid_label[j, ...]
            i += 1
        x_batch = TINY_IMAGENET200.preprocessData(x_batch)
        return x_batch, y_batch
                #yield x_batch, y_batch
