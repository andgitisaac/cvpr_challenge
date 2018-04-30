from __future__ import print_function, division

import os
import numpy as np
import skimage.io
import scipy.ndimage
from PIL import Image
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
                                    save_best_only=False,
                                    save_weights_only=True,
                                    period=2,
                                    mode='auto')

        self.model.train(batch_size=self.batch_size,
                        epochs=self.epochs,
                        steps=self.steps,
                        train_path=self.train_path,
                        val_path=self.val_path,
                        cp_cb=self.cp_cb)
    
    def eval(self, output_mode):
        # print('loading model from {}...'.format(self.model.model_path))        
        # self.model.load(self.model.model_path)
        # print('Model is loaded!!!')
        def merge(group, size=(612, 612), num_per_side=13):
            np_mask = np.zeros((2448, 2448, 7))
            # img = Image.new('RGB', (2448, 2448))
            for idx, chunk in enumerate(group):
                offset_x = idx // num_per_side * size[0] // 4
                offset_y = idx % num_per_side * size[1] // 4
                np_mask[offset_x:offset_x+size[0], offset_y:offset_y+size[1], :] = np_mask[offset_x:offset_x+size[0], offset_y:offset_y+size[1], :] + chunk[:, :, :]
            img = mask_postprocess(np_mask)
            img = Image.fromarray(img)
            return img


        self.model.build_model()
        content_path, _ = get_file_paths(self.test_path)
        content_path.sort()


        group = []

        for cnt, path in enumerate(content_path):
            # assert len(content_path) % 169 == 0

            
            img = np.expand_dims(vgg_sub_mean(get_image(path)), axis=0)
            reconst_mask = self.model.decode(img)

            if output_mode == 'img':
                fileID = path.split('/')[-1].split('_')[0]
                output_name = '{}_mask.png'.format(fileID)
                output_name = os.path.join(self.output_path, output_name)
                print("({}/{}) {}".format(cnt, len(content_path), output_name))

                reconst_mask = mask_postprocess(reconst_mask[0])
                reconst_mask = image_resize(reconst_mask, size=(612, 612))
                skimage.io.imsave(output_name, reconst_mask)
            elif output_mode == 'npz':
                if cnt % 169 == 0 and cnt != 0:
                    print('Saving {}'.format(output_name))
                    img = merge(group)
                    img.save(output_name)
                    group = []

                fileID = path.split('/')[-1].split('-')[0]
                output_name = '{}_mask.png'.format(fileID)
                output_name = os.path.join(self.output_path, output_name)
                print("({}/{}) {}".format(cnt+1, len(content_path), output_name))

                scale = 612./512.
                reconst_mask = scipy.ndimage.interpolation.zoom(reconst_mask[0],
                                                                zoom=(scale, scale, 1),
                                                                mode='reflect')
                
                
                
                group.append(reconst_mask)
        
        # Saving the last figure
        if output_mode == 'npz':
            print('Saving {}'.format(output_name))
            img = merge(group)
            img.save(output_name)
                

