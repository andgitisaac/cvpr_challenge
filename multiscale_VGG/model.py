from __future__ import print_function, division

import random
import numpy as np
from keras.utils import plot_model
from keras.layers import Input, Lambda, Conv2D, Conv2DTranspose, MaxPooling2D, Activation, Add
from keras.models import Model
from keras.optimizers import SGD
from BilinearUpSampling import *
from utils import get_file_paths, get_image, vgg_sub_mean, mask_preprocess, random_crop

class MultiscaleVGG(object):

    def __init__(self,
                batch_shape=(512, 512, 3),
                scale=None,
                mode='train',
                import_model=None):

        self.mode = mode
        self.batch_shape = batch_shape
        self.scale = scale
        self.import_model = import_model

    def build_model(self):

        ### Build Modified VGG-16 ###
        content_input = Input(shape=self.batch_shape)

        # Block 1
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(content_input)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
        pool1_1 = x

        # Block 2
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
        pool2_1 = x 

        # Block 3
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
        pool3_1 = x

        # Block 4
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
        # x = MaxPooling2D((2, 2), strides=(1, 1), padding='same', name='block4_pool')(x)
        pool4_1 = x

        # Block 5
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
        # x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
        x = MaxPooling2D((2, 2), strides=(1, 1), padding='same', name='block5_pool')(x)
        pool5_1 = x

        self.vgg = Model(input=content_input, output=pool5_1)

        ### Build UpStream_1 ###
        x = self.feature_layer(_input=pool5_1, prefix='upstream1')
        x = Conv2DTranspose(7, (32, 32), strides=(16, 16), padding='same', use_bias=False, name='upstream1_up')(x)
        # x = BilinearUpSampling2D(size=(8, 8), name='upstream1_up')(x)
        x = Activation('softmax', name='upstream1_output')(x)

        self.upstream1 = Model(input=content_input, output=x)

        ### Build UpStream_2 ###
        x = self.feature_layer(_input=pool4_1, prefix='upstream2')
        x = Conv2DTranspose(7, (32, 32), strides=(16, 16), padding='same', use_bias=False, name='upstream2_up')(x)
        # x = BilinearUpSampling2D(size=(8, 8), name='upstream2_up')(x)
        x = Add(name='upstream2_add')([x, self.upstream1.layers[-2].output])
        x = Activation('softmax', name='upstream2_output')(x)

        self.upstream2 = Model(input=content_input, output=x)

        ### Build UpStream_3 ###
        x = self.feature_layer(_input=pool3_1, prefix='upstream3')
        x = Conv2DTranspose(7, (16, 16), strides=(8, 8), padding='same', use_bias=False, name='upstream3_up')(x)
        # x = BilinearUpSampling2D(size=(8, 8), name='upstream3_up')(x)
        x = Add(name='upstream3_add')([x, self.upstream2.layers[-2].output])
        x = Activation('softmax', name='upstream3_output')(x)

        self.upstream3 = Model(input=content_input, output=x)

        ### Build UpStream_4 ###
        x = self.feature_layer(_input=pool2_1, prefix='upstream4')
        x = Conv2DTranspose(7, (8, 8), strides=(4, 4), padding='same', use_bias=False, name='upstream4_up')(x)
        # x = BilinearUpSampling2D(size=(4, 4), name='upstream4_up')(x)
        x = Add(name='upstream4_add')([x, self.upstream3.layers[-2].output])
        x = Activation('softmax', name='upstream4_output')(x)

        self.upstream4 = Model(input=content_input, output=x)

        ### Build UpStream_5 ###
        x = self.feature_layer(_input=pool1_1, prefix='upstream5')
        x = Conv2DTranspose(7, (4, 4), strides=(2, 2), padding='same', use_bias=False, name='upstream5_up')(x)
        # x = BilinearUpSampling2D(size=(2, 2), name='upstream5_up')(x)
        x = Add(name='upstream5_add')([x, self.upstream4.layers[-2].output])
        x = Activation('softmax', name='upstream5_output')(x)

        self.upstream5 = Model(input=content_input, output=x)


        model_dict = [self.upstream1, self.upstream2,
                    self.upstream3, self.upstream4, self.upstream5]
        

        self.model = model_dict[self.scale-1]
        
        if self.mode == 'train':
            print('Loading VGG16 weights...')
            self.vgg.load_weights(self.import_model, by_name=True)
            for layer in self.vgg.layers[:-4]:
                layer.trainable = False
            print('VGG16 is loaded!')
            # # The coarest layer only needs to load weights of vgg16
            # if self.scale == 1:
            #     print('Loading VGG16 weights...')
            #     self.vgg.load_weights(self.import_model, by_name=True)

            #     # block4, block5 in modified vgg need to be trained
            #     for layer in self.vgg.layers[:-8]:
            #         layer.trainable = False
            #     print('VGG16 is loaded!')
            # else:
            #     print('Loading weights of UP_STREAM_{}...'.format(self.scale-1))
            #     model_dict[self.scale-2].load_weights(self.import_model, by_name=True)
            #     for layer in model_dict[self.scale-1].layers:                    
            #         if ('upstream{}'.format(self.scale-1) not in layer.get_config()['name']) and ('upstream{}'.format(self.scale) not in layer.get_config()['name']):                                       
            #             layer.trainable = False
            #         else:
            #             print("{} is trainable".format(layer.get_config()['name']))                        
            #     for layer in self.vgg.layers[-8:]:
            #         layer.trainable = True
            #         print("{} is trainable".format(layer.get_config()['name']))
            #     print('Model UPSTREAM {} is loaded!'.format(self.scale-1))

        elif self.mode == 'eval':
            pass
                            
        return self.model

    def feature_layer(self, _input, prefix):
        x = Conv2D(1024, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name=prefix+'_conv1')(_input)
        x = Conv2D(1024, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', name=prefix+'_conv2')(x)
        x = Conv2D(7, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal', name=prefix+'_conv3')(x)
        return x

    def batch_gen(self, path, batch_size, crop_size):
        content_path, mask_path = get_file_paths(path)
        
        while True:
            index = random.sample(range(1, len(content_path)), batch_size)
            try:
                offset_h = random.randint(0, (2448-crop_size[0]))
                offset_w = random.randint(0, (2448-crop_size[1]))
                offset = (offset_h, offset_w)
                
                contents = [vgg_sub_mean(random_crop(get_image(content_path[i]), offset, crop_size)) for i in index]
                masks = [mask_preprocess(random_crop(get_image(mask_path[i]), offset, crop_size)) for i in index]

                contents = np.asarray(contents, dtype=np.float32)
                masks = np.asarray(masks, dtype=np.uint8)

            except Exception as err:
                print("\nError: {}".format(err))
                continue

            yield contents, masks

    def compile(self, learning_rate):
        # self.optimizer = SGD(lr=learning_rate, momentum=0.9)
        self.optimizer = 'adadelta'
        self.model.compile(optimizer=self.optimizer,
                            loss='categorical_crossentropy',
                            metrics=['accuracy'])
    
    def train(self, batch_size, crop_size, epochs, steps, train_path, val_path, cp_cb):
        self.model.fit_generator(self.batch_gen(train_path, batch_size, crop_size),
                                epochs=epochs,
                                steps_per_epoch=steps,
                                validation_data=self.batch_gen(val_path, batch_size, crop_size),
                                validation_steps=50,
                                verbose=1,
                                callbacks=[cp_cb])
    
    def decode(self, x):
        return self.model.predict(x)

    def summary(self):
        self.model.summary()
    
    def plot_model(self):
        plot_model(self.model, tofile='multiscale_vgg.png')

