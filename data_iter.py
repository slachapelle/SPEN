# pylint: skip-file
""" file iterator for denoising dataset"""
import mxnet as mx
import numpy as np
import sys, os
from mxnet.io import DataIter
from PIL import Image
# import cv2
import random


class FileIter(DataIter):
    """FileIter object in fcn-xs example. Taking a file list file to get dataiter.
    in this example, we use the whole image training for fcn-xs, that is to say
    we do not need resize/crop the image to the same size, so the batch_size is
    set to 1 here
    Parameters
    ----------
    root_dir : string
        the root dir of image/label lie in
    flist_name : string
        the list file of iamge and label, every line owns the form:
        index \t image_data_path \t image_label_path
    cut_off_size : int
        if the maximal size of one image is larger than cut_off_size, then it will
        crop the image with the minimal size of that image
    data_name : string
        the data name used in symbol data(default data name)
    label_name : string
        the label name used in symbol softmax_label(default label name)
    """
    def __init__(self, root_dir, flist_name,
                 rgb_mean = (0.41, 0.41, 0.41),
                 cut_off_size = None,
                 batch_size = 8,
                 data_name = "data",
                 label_name = "label"):
        super(FileIter, self).__init__()
        self.root_dir = root_dir
        self.flist_name = os.path.join(self.root_dir, flist_name)
        self.mean = np.array(rgb_mean)  # (R, G, B)
        self.cut_off_size = cut_off_size
        self.data_name = data_name
        self.label_name = label_name
        self.batch_size = batch_size
        self.num_data = len(open(self.flist_name, 'r').readlines())
        self.f = open(self.flist_name, 'r')
        self.data, self.label = self._read()
        self.cursor = -1

    def _read(self):
        """get two list, each list contains two elements: name and nd.array value"""
        data = {}
        label = {}

        _, data_img_name, label_img_name = self.f.readline().strip('\n').split(" ")
        data[self.data_name], label[self.label_name] = self._read_img(data_img_name, label_img_name)
        for k in range(self.batch_size - 1):
            _, data_img_name, label_img_name = self.f.readline().strip('\n').split(" ")
            temp_data, temp_label = self._read_img(data_img_name, label_img_name)
            data[self.data_name] = np.concatenate((data[self.data_name], temp_data), axis=0)
            label[self.label_name] = np.concatenate((label[self.label_name], temp_label), axis=0)
        print(data[self.data_name].shape)
        return list(data.items()), list(label.items())

    def _read_img(self, img1_name, label_name):

        data_buffer = np.fromfile(os.path.join(self.root_dir, img1_name), dtype=np.int16)
        noisy = data_buffer[2:]
        noisy = noisy.reshape((data_buffer[0], data_buffer[1]))
        noisy = np.clip(noisy.astype(np.float32)/5000, 0, 1)

        data_buffer = np.fromfile(os.path.join(self.root_dir, label_name), dtype=np.int16)
        clean = data_buffer[2:]
        clean = clean.reshape((data_buffer[0], data_buffer[1]))
        clean = np.clip(clean.astype(np.float32)/5000, 0, 1)
        sz = clean.shape
        if (sz[0] > sz[1]):
            clean = np.swapaxes(clean, 0, 1)
            noisy = np.swapaxes(noisy, 0, 1)

        if (random.uniform(0, 1.0) > 0.5):
            clean = np.fliplr(clean)
            noisy = np.fliplr(noisy)

        # print(clean.shape)

        left = int(random.uniform(0, 214-129))
        right = 128 + left
        up = int(random.uniform(0, 160-97))
        down = 96 + up
        clean = clean[up:down,left:right]
        noisy = noisy[up:down,left:right]

        # print(clean.shape)

        clean = np.expand_dims(clean, axis=2)  # (1, c, h, w)
        clean = np.swapaxes(clean, 0, 2)
        clean = np.swapaxes(clean, 1, 2)  # (c, h, w)
        clean = np.expand_dims(clean, axis=0)  # (1, c, h, w)


        noisy = np.expand_dims(noisy, axis=2)  # (1, c, h, w)
        noisy = np.swapaxes(noisy, 0, 2)
        noisy = np.swapaxes(noisy, 1, 2)  # (c, h, w)
        noisy = np.expand_dims(noisy, axis=0)  # (1, c, h, w)
        return (noisy, clean)

    @property
    def provide_data(self):
        """The name and shape of data provided by this iterator"""
        return [(k, tuple([1] + list(v.shape[1:]))) for k, v in self.data]

    @property
    def provide_label(self):
        """The name and shape of label provided by this iterator"""
        return [(k, tuple([1] + list(v.shape[1:]))) for k, v in self.label]

    def get_batch_size(self):
        return self.batch_size

    def reset(self):
        self.cursor = -1
        self.f.close()
        self.f = open(self.flist_name, 'r')

    def iter_next(self):
        self.cursor += self.batch_size
        if(self.cursor < self.num_data-self.batch_size-1):
            return True
        else:
            return False

    def next(self):
        """return one dict which contains "data" and "label" """
        if self.iter_next():
            self.data, self.label = self._read()
            return {self.data_name  :  self.data[0][1],
                    self.label_name :  self.label[0][1]}
        else:
            raise StopIteration