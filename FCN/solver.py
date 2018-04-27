from __future__ import print_function, division

import os
import numpy as np
import skimage.io
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from utils import get_file_paths, get_image, vgg_sub_mean, mask_preprocess, mask_postprocess, image_resize, mean_IoU

class Solver(object):

    def __init__(self, model, batch_size=3, epochs=20,
                steps=None, learning_rate=1e-4,
                train_path=None, val_path=None,
                test_path=None, output_path=None):

        self.model = model
        self.batch_size = batch_size
        self.epochs = epochs
        self.steps = steps
        self.learning_rate = learning_rate
        self.train_path = train_path
        self.val_path = val_path if val_path != None else train_path
        self.test_path = test_path
        self.output_path = output_path



    def train(self):
        self.model.build_model()
        self.model.summary()
        self.model.compile(self.learning_rate)

        self.chkpt = 'fcn32_model/VGG_FCN.{epoch:02d}-[{mean_IoU:.03f}].h5'
        self.cp_cb = ModelCheckpoint(filepath=self.chkpt,
                                    monitor='val_loss',
                                    verbose=1,
                                    save_best_only=True,
                                    mode='auto')

        self.model.train(batch_size=self.batch_size,
                        epochs=self.epochs,
                        steps=self.steps,
                        train_path=self.train_path,
                        val_path=self.val_path,
                        cp_cb=self.cp_cb)
    
    def eval(self):
        # print('loading model from {}...'.format(self.model.model_path))        
        # self.model.load(self.model.model_path)
        # print('Model is loaded!!!')
        self.model.build_model()
        content_path, _ = get_file_paths(self.test_path)
        content_path.sort()

        for path in content_path:
            fileID = path.split('/')[-1].split('_')[0]
            output_name = '{}_mask.png'.format(fileID)
            output_name = os.path.join(self.output_path, output_name)
            print(output_name)
            img = np.expand_dims(vgg_sub_mean(get_image(path)), axis=0)
            reconst_mask = self.model.decode(img)
            reconst_mask = mask_postprocess(reconst_mask[0])
            reconst_mask = image_resize(reconst_mask, size=(612, 612))
            skimage.io.imsave(output_name, reconst_mask)