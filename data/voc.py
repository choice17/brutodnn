import numpy as np
import xml.etree.ElementTree as ET
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from utils.utils import imshow

voc_path = 'data/dataset/VOCdevkit/'

voc_2012_image_folder = 'VOC2012/JPEGImages/'
voc_2012_annot_folder = 'VOC2012/Annotations/'
voc_2012_train_image_folder = 'VOC2012/ImageSets/Main/train.txt'
voc_2012_valid_image_folder = 'VOC2012/ImageSets/Main/val.txt'

voc_2007_image_folder = 'VOC2007/JPEGImages/'
voc_2007_annot_folder = 'VOC2007/Annotations/'
voc_2007_train_image_folder = 'VOC2007/ImageSets/Main/train.txt'
voc_2007_valid_image_folder = 'VOC2007/ImageSets/Main/val.txt'
voc_2007_test_image_folder = 'VOC2007/ImageSets/Main/test.txt'

voc_class = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 
'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
'sofa', 'train', 'tvmonitor']

voc_class_key = {key:item for item, key in enumerate(voc_class)}
voc_class_idx = np.zeros((len(voc_class)))

VOC_2007 = 0
VOC_2012 = 1
VOC_ALL  = 2

BATCH_SIZE = 4
NUM_CLASSES = 20
EPOCHS = 1
DATASET = "."
INPUT_DIM = (227, 227, 3)

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

class VOC2007(object):
    image_folder = voc_2007_image_folder
    annot_folder = voc_2007_annot_folder
    train_image_folder = voc_2007_train_image_folder
    valid_image_folder = voc_2007_valid_image_folder
    test_image_folder  = voc_2007_test_image_folder
    trainval_img_folder = [train_image_folder, valid_image_folder] 
    test_img_folder = [test_image_folder]

class VOC2012(object):
    image_folder = voc_2012_image_folder
    annot_folder = voc_2012_annot_folder
    train_image_folder = voc_2012_train_image_folder
    valid_image_folder = voc_2012_valid_image_folder
    trainval_img_folder = [ train_image_folder, valid_image_folder ]

class VOC(object):

    datagen = ImageDataGenerator(
                featurewise_center=False,
                samplewise_center=False,
                featurewise_std_normalization=False,
                samplewise_std_normalization=False,
                zca_whitening=False,
                zca_epsilon=1e-6,
                rotation_range=30,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.3,
                zoom_range=0.1,
                channel_shift_range=30,
                fill_mode='nearest',
                cval=0.,
                horizontal_flip=True,
                vertical_flip=False,
                rescale=None,
                preprocessing_function=None)

    VOC_2007 = VOC_2007
    VOC_2012 = VOC_2012
    VOC_ALL  = VOC_ALL

    def __init__(self, voc_data= VOC_2012):
        self.voc_class = voc_class
        self.dirs = {'voc_path': voc_path }

        if voc_data == VOC_2012:
            self.train_image_folder = [VOC2012.train_image_folder]
            self.valid_image_folder = [VOC2012.valid_image_folder]
        elif voc_data == VOC_2007:
            self.train_image_folder = VOC2007.trainval_img_folder
            self.valid_image_folder = VOC2007.test_img_folder
        elif voc_data == VOC_ALL:
            self.train_image_folder = VOC2007.trainval_img_folder + VOC2012.trainval_img_folder
            self.valid_image_folder = VOC2007.test_img_folder


    def set(self, voc_path):
        self.dirs['voc_path'] = voc_path

        if not os.path.exists(self.dir['voc_path']):
            print("Cannot find dataset from %s", voc_path)
            exit()

    def getFileList(self):
        print("[INFO] Loading File list ...")
        self.train_img_list = []
        self.train_annot_list = []

        for filepath in self.train_image_folder:
            filepath = self.dirs['voc_path'] + filepath
            with  open(filepath, 'r') as f:
                img_list = f.read().split('\n')
            if filepath.find('2012')>=0:
                image_folder = VOC2012.image_folder
                annot_folder = VOC2012.annot_folder
            else:
                image_folder = VOC2007.image_folder
                annot_folder = VOC2007.annot_folder
            print(filepath, image_folder,  annot_folder)
            for l in img_list:
                if l:
                    img_path = '%s%s%s.jpg' % (self.dirs['voc_path'],
                            image_folder, l)
                    self.train_img_list.append(img_path)
                    annot_path = '%s%s%s.xml' % (self.dirs['voc_path'],
                            annot_folder, l)
                    self.train_annot_list.append(annot_path)

        self.valid_img_list = []
        self.valid_annot_list = []

        for filepath in self.valid_image_folder:
            filepath = self.dirs['voc_path'] + filepath
            with  open(filepath, 'r') as f:
                img_list = f.read().split('\n')
            if filepath.find('2012')>=0:
                image_folder = VOC2012.image_folder
                annot_folder = VOC2012.annot_folder
            else:
                image_folder = VOC2007.image_folder
                annot_folder = VOC2007.annot_folder
            print(filepath, image_folder,  annot_folder)
            for l in img_list:
                if l:
                    img_path = '%s%s%s.jpg' % (self.dirs['voc_path'],
                            image_folder, l)
                    self.valid_img_list.append(img_path)
                    annot_path = '%s%s%s.xml' % (self.dirs['voc_path'],
                            annot_folder, l)
                    self.valid_annot_list.append(annot_path)

    def getClassAnnot(self):
        print("[INFO] Getting Class annot  ...")
        _len = len(self.train_annot)
        self.train_label = np.zeros((_len, len(voc_class)))
        j = 0
        for i in self.train_annot:
            objs = i.findall('object')
            for obj in objs:
                self.train_label[j, voc_class_key[obj[0].text]] = 1
            j += 1

        _len = len(self.valid_annot)
        self.valid_label = np.zeros((_len, len(voc_class)))
        j = 0
        for i in self.valid_annot:
            objs = i.findall('object')
            for obj in objs:
                self.valid_label[j, voc_class_key[obj[0].text]] = 1
            j += 1

    def getAnnot(self):
        print("[INFO] Load annot from list ...")
        self.train_annot = []
        for i in self.train_annot_list:
            f = open(i, 'r')
            self.train_annot.append(ET.fromstring(f.read()))
            f.close()

        self.valid_annot = []
        for i in self.valid_annot_list:
            f = open(i, 'r')
            self.valid_annot.append(ET.fromstring(f.read()))
            f.close()

    def setGeneratorConfig(self, config=config):
        """
        batch_size : 64
        img_h : 224
        img_w : 224
        img_c : 3
        norm : 1
        num_classes: 20
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
        return VOC_TRAIN_BATCH(self)

    def getValidBatch(self):
        if self.config['data_mode'] == 'on_memory':
            return self.valid_images, self.valid_label
        return VOC_VAL_BATCH(self)

    def preprocessData(img):
        img = img.astype(np.float32) / 255 - 0.5
        return img
    
    def augment_data(img_batch, y_batch, batch_size):
        img_batch, y_batch = VOC.datagen.flow((img_batch, y_batch), batch_size=batch_size).next()
        return img_batch, y_batch

class VOC_TRAIN_BATCH(Sequence):

    def __init__(self, voc):
        self.config = voc.config
        self.train_label = voc.train_label
        self.train_img_list = voc.train_img_list
        self.shuffle = voc.config['shuffle']
        self.generate_list = list(range(len(self)))
        self.augment_data = 0
        print("[INFO] total %d training images" % len(voc.train_img_list))

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
            x_batch, y_batch = VOC.augment_data(x_batch, y_batch, batch_size=self.config['batch_size'])
        x_batch = VOC.preprocessData(x_batch)
        return x_batch, y_batch

class VOC_VAL_BATCH(Sequence):

    def __init__(self, voc):
        self.config = voc.config
        self.valid_label = voc.valid_label
        self.valid_img_list = voc.valid_img_list
        self.shuffle = voc.config['shuffle']
        self.generate_list = list(range(len(self)))
        print("[INFO] total %d validation images" % len(voc.valid_img_list))

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
        x_batch = VOC.preprocessData(x_batch)
        return x_batch, y_batch
                #yield x_batch, y_batch
