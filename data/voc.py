import numpy as np
import xml.etree.ElementTree as ET
from tensorflow.keras.utils import Sequence
import cv2

voc_path = 'data/dataset/VOCdevkit/'
voc_2012_image_folder = 'VOC2012/JPEGImages/'
voc_2012_train_image_folder = 'VOC2012/ImageSets/Main/train.txt'
voc_2012_train_annot_folder = 'VOC2012/Annotations/'
voc_2012_valid_image_folder = 'VOC2012/ImageSets/Main/val.txt'
voc_2012_valid_annot_folder = 'VOC2012/Annotations/'

voc_class = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 
'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
'sofa', 'train', 'tvmonitor']

voc_class_key = {key:item for item, key in enumerate(voc_class)}
voc_class_idx = np.zeros((len(voc_class)))

class VOC(object):

    def __init__(self, *args, **kwargs):
        super(VOC, self).__init__(*args, **kwargs)
        self.voc_clas = voc_class
        self.dirs = {
            'voc_path': voc_path,
            'voc_2012_image_folder': voc_2012_image_folder,
            'voc_2012_train_image_folder': voc_2012_train_image_folder,
            'voc_2012_train_annot_folder': voc_2012_valid_annot_folder,
            'voc_2012_valid_annot_folder': voc_2012_valid_annot_folder,
            'voc_2012_valid_image_folder': voc_2012_valid_image_folder
            }

    def set(self, voc_path):
        self.dirs['voc_path'] = voc_path

    def getFileList(self):
        filepath = self.dirs['voc_path'] + self.dirs['voc_2012_train_image_folder']
        with  open(filepath, 'r') as f:
            img_list = f.read().split('\n')
        self.train_img_list = []
        self.train_annot_list = []
        for l in img_list:
            if l:
                img_path = '%s%s%s.jpg' % (self.dirs['voc_path'],
                        self.dirs['voc_2012_image_folder'], l)
                self.train_img_list.append(img_path)
                annot_path = '%s%s%s.xml' % (self.dirs['voc_path'],
                        self.dirs['voc_2012_train_annot_folder'], l)
                self.train_annot_list.append(annot_path)

        filepath = self.dirs['voc_path'] + self.dirs['voc_2012_valid_image_folder']
        with  open(filepath, 'r') as f:
            img_list = f.read().split('\n')
        self.valid_img_list = []
        self.valid_annot_list = []
        for l in img_list:
            if l:
                img_path = '%s%s%s' % (self.dirs['voc_path'],
                        self.dirs['voc_2012_image_folder'], l)
                self.valid_img_list.append(img_path)
                annot_path = '%s%s%s.xml' % (self.dirs['voc_path'],
                        self.dirs['voc_2012_valid_annot_folder'], l)
                self.valid_annot_list.append(annot_path)

    def getClassAnnot(self):
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

    def setGeneratorConfig(self, config):
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

    def getTrainBatch(self):
        return VOC_TRAIN_BATCH(self)

    def getValidBatch(self):
        return VOC_VAL_BATCH(self)

"""
    def __len__(self):
        return int((self.train_label.shape[0] / self.config['batch_size']) + 0.5)

    def size(self):
        return self.train_label.shape[0]

    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.generate_list)

    def __getitem__(self, *args, **kwargs):
        x_batch = np.empty((self.config['batch_size'],
                            self.config['img_h'],
                            self.config['img_w'],
                            self.config['img_c']))
        y_batch = np.empty((self.config['batch_size'],
                            self.config['num_class']))
        self.generate_list = list(range(len(self)))

        while True:
            for idx in self.generate_list:
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
                    x_batch[i, ...] = cv2.resize(img, (self.config['img_w'], self.config['img_h']))
                    y_batch[i, ...] = self.train_label[j, ...]
                    i += 1
                yield x_batch, y_batch
"""

class VOC_TRAIN_BATCH(Sequence):

    def __init__(self, voc):
        self.config = voc.config
        self.train_label = voc.train_label
        self.train_img_list = voc.train_img_list
        self.shuffle = voc.config['shuffle']
        self.generate_list = list(range(len(self)))

    def __len__(self):
        return int((self.train_label.shape[0] / self.config['batch_size']) + 0.5)

    def size(self):
        return self.train_label.shape[0]

    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.generate_list)

    def __getitem__(self, idx):
        x_batch = np.empty((self.config['batch_size'],
                            self.config['img_h'],
                            self.config['img_w'],
                            self.config['img_c']))
        y_batch = np.empty((self.config['batch_size'],
                            self.config['num_class']))
 
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
            x_batch[i, ...] = cv2.resize(img, (self.config['img_w'], self.config['img_h']))
            y_batch[i, ...] = self.train_label[j, ...]
            i += 1
        return x_batch, y_batch

        """
        x_batch = np.empty((self.config['batch_size'],
                            self.config['img_h'],
                            self.config['img_w'],
                            self.config['img_c']))
        y_batch = np.empty((self.config['batch_size'],
                            self.config['num_class']))
        while True:
            for idx in self.generate_list:
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
                    x_batch[i, ...] = cv2.resize(img, (self.config['img_w'], self.config['img_h']))
                    y_batch[i, ...] = self.train_label[j, ...]
                    i += 1
                yield x_batch, y_batch
        """

class VOC_VAL_BATCH(Sequence):

    def __init__(self, voc):
        self.config = voc.config
        self.valid_label = voc.valid_label
        self.valid_img_list = voc.valid_img_list
        self.shuffle = voc.config['shuffle']
        self.generate_list = list(range(len(self)))

    def __len__(self):
        return int((self.valid_label.shape[0] / self.config['batch_size']) + 0.5)

    def size(self):
        return self.valid_label.shape[0]

    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.generate_list)

    def __getitem__(self, *args, **kwargs):
        x_batch = np.empty((self.config['batch_size'],
                            self.config['img_h'],
                            self.config['img_w'],
                            self.config['img_c']))
        y_batch = np.empty((self.config['batch_size'],
                            self.config['num_class']))
        while True:
            for idx in self.generate_list:
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
                    x_batch[i, ...] = cv2.resize(img, (self.config['img_w'], self.config['img_h']))
                    y_batch[i, ...] = self.valid_label[j, ...]
                    i += 1
                yield x_batch, y_batch












