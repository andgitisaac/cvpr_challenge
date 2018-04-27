from __future__ import print_function, division

import random
import glob
import skimage.io
from PIL import Image
import numpy as np
import keras.backend as K
import tensorflow as tf


color2index = {
    (0  , 255, 255) : 0,
    (255, 255,   0) : 1,
    (255,   0, 255) : 2,
    (0  , 255,   0) : 3,
    (  0,   0, 255) : 4,
    (255, 255, 255) : 5,
    (  0,   0,   0) : 6
}

index2color = {
    0 : (0  , 255, 255),
    1 : (255, 255,   0),
    2 : (255,   0, 255),
    3 : (0  , 255,   0),
    4 : (  0,   0, 255),
    5 : (255, 255, 255),
    6 : (  0,   0,   0)
}

vgg_mean = [103.939, 116.779, 123.68]

### image reading ###
def get_file_paths(dir):
    content = glob.glob("{}/*.jpg".format(dir))
    mask = glob.glob("{}/*.png".format(dir))
    return content, mask

def get_image(path):
    img = skimage.io.imread(path)
    assert len(img.shape) == 3, "# of channels of {} is not 3".format(path)
    return img

def vgg_sub_mean(img):
    img = img.astype(np.float32)
    for i in range(3):
        img[:, :, i] -= vgg_mean[i] # int ??
    return img

def image_resize(img, size):
    # print(img.shape)
    img = Image.fromarray(img)
    img = img.resize(size, resample=Image.BILINEAR)
    # print(img.size)
    return img


def random_crop(img, offset, crop=(512, 512)):
    img = img[offset[0]:offset[0]+crop[0], offset[1]:offset[1]+crop[1], :]
    return img

### mask processing ###
def mask_preprocess(img, n_classes=7):
    label = np.ndarray(shape=img.shape[:2], dtype=np.uint8)
    label[:, :] = -1
    for rgb, idx in color2index.items():
        label[(img == rgb).all(2)] = idx
    one_hot = np.eye(n_classes, dtype=np.uint8)[label]
    return one_hot

def mask_postprocess(one_hot):
    predict = np.argmax(one_hot, axis=-1)
    height, width = predict.shape[:2]
    mask = np.zeros((height, width, 3), dtype=np.uint8)

    for h in range(height):
        for w in range(width):
            mask[h, w] = index2color[predict[h, w]]
    return mask

### metrics ###
def mean_IoU(y_true, y_pred):
    s = K.shape(y_true)

	# reshape such that w and h dim are multiplied together
    y_true_reshaped = K.reshape( y_true, tf.stack( [-1, s[1]*s[2], s[-1]] ) )
    y_pred_reshaped = K.reshape( y_pred, tf.stack( [-1, s[1]*s[2], s[-1]] ) )

    # correctly classified
    clf_pred = K.one_hot( K.argmax(y_pred_reshaped), num_classes = s[-1])
    equal_entries = K.cast(K.equal(clf_pred,y_true_reshaped), dtype='float32') * y_true_reshaped

    intersection = K.sum(equal_entries, axis=1)
    union_per_class = K.sum(y_true_reshaped,axis=1) + K.sum(y_pred_reshaped,axis=1)

    iou = intersection / (union_per_class - intersection)
    iou_mask = tf.is_finite(iou)
    iou_masked = tf.boolean_mask(iou,iou_mask)

    return K.mean( iou_masked )
    






