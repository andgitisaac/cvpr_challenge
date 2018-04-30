from __future__ import print_function, division

import random
import numpy as np
from keras.utils import plot_model
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Dropout, Activation, Conv2DTranspose, Cropping2D, Add, ZeroPadding2D
from keras.models import Model
from keras.optimizers import SGD
from utils import get_file_paths, get_image, vgg_sub_mean, mask_preprocess, random_crop, mean_IoU

class VGG_FCN(object):

    def __init__(self, batch_shape=(512, 512, 3), mode='train',
                vgg_path=None, model_path=None):

        self.mode = mode
        self.batch_shape = batch_shape
        self.vgg_path = vgg_path
        self.model_path = model_path

    def build_model(self):

        ### Build VGG-16 ###
        content_input = Input(shape=self.batch_shape)

        # Block 1
        conv1_1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(content_input)
        conv1_2 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(conv1_1)
        pool1_1 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(conv1_2)

        # Block 2
        conv2_1 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(pool1_1)
        conv2_2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(conv2_1)
        pool2_1 = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(conv2_2)

        # Block 3
        conv3_1 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(pool2_1)
        conv3_2 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(conv3_1)
        conv3_3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(conv3_2)
        pool3_1 = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(conv3_3)

        # Block 4
        conv4_1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(pool3_1)
        conv4_2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(conv4_1)
        conv4_3 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(conv4_2)
        pool4_1 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(conv4_3)

        # Block 5
        conv5_1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(pool4_1)
        conv5_2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(conv5_1)
        conv5_3 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(conv5_2)
        pool5_1 = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(conv5_3)

        self.vgg = Model(input=content_input, output=pool5_1)

        ### Build FCN-8s ###
        
        # Fully-Connected layers
        conv6 = Conv2D(4096, (7, 7), activation='relu', padding='same', name='fcn_fc1')(pool5_1)
        drop6 = Dropout(0.5)(conv6)
        conv7 = Conv2D(4096, (1, 1), activation='relu', padding='same', name='fcn_fc2')(drop6)
        drop7 = Dropout(0.5)(conv7)

        # Classifier
        # conv8 = Conv2D(7, (1, 1), strides=(1, 1), kernel_initializer='he_normal', activation='linear', padding='valid')(drop7)
        # up8 = Conv2DTranspose(7, (64, 64), strides=(32, 32), padding='same', name='deconv')(conv8)
        # up8 = Conv2DTranspose(7, (32, 32), strides=(16, 16), padding='same', use_bias=False, name='deconv9_1')(conv8)
        # pad9 = ZeroPadding2D(padding=(1, 1), name='padding9')(up8)
        # up10 = Conv2DTranspose(7, (4, 4), strides=(2, 2), padding='same', use_bias=False, name='deconv9_2')(pad9)
        # output = Activation('softmax')(up10)
        
        conv8 = Conv2D(7, (1, 1), strides=(1, 1), kernel_initializer='he_normal', activation='linear', padding='valid', name='conv8')(drop7)
        up8 = Conv2DTranspose(7, (4, 4), strides=(2, 2), padding='same', use_bias=False, name='deconv8_1')(conv8)
        
        scaled_pool4 = Lambda(lambda x: x * 0.01, name='scaled_pool4')(pool4_1)      
        conv9 = Conv2D(7, (1, 1), strides=(1, 1), kernel_initializer='he_normal', padding='valid', name='conv9')(scaled_pool4)
        o1, o2 = self.crop(up8, conv9, content_input)
        add9 = Add()([o1, o2])
        up9 = Conv2DTranspose(7, (4, 4), strides=(2, 2), padding='same', use_bias=False, name='deconv9_1')(add9)

        scaled_pool3 = Lambda(lambda x: x * 0.0001, name='scaled_pool3')(pool3_1)  
        conv10 = Conv2D(7, (1, 1), strides=(1, 1), kernel_initializer='he_normal', padding='valid', name='conv10')(scaled_pool3)
        o1, o2 = self.crop(up9, conv10, content_input)
        add10 = Add()([o1, o2])

        up11 = Conv2DTranspose(7, (16, 16), strides=(8, 8), padding='same', use_bias=False, name='deconv10_1')(add10)
        output = Activation('softmax')(up11)

        self.model = Model(input=content_input, output=output)
        
        if self.mode == 'train':
            print('Loading VGG16 weights...')
            self.vgg.load_weights(self.vgg_path, by_name=True)
            for layer in self.vgg.layers:
                layer.trainable = False
            print('VGG16 weights loaded!')
        elif self.mode == 'eval':
            print('Loading model from {}...'.format(self.model_path))
            self.model.load_weights(self.model_path, by_name=True)
            print('Model weights loaded!')
                            
        return self.model

    def crop(self, o1, o2, input):
        o1_shape = Model(input=input, output=o1).output_shape
        o1_height = o1_shape[1]
        o1_width = o1_shape[2]

        o2_shape = Model(input=input, output=o2).output_shape
        o2_height = o2_shape[1]
        o2_width = o2_shape[2]

        cx = abs(o1_width - o2_width)
        cy = abs(o1_height - o2_height)

        if (o1_width > o2_width):
            o1 = Cropping2D(cropping=((0, 0) ,  (0, cx)))(o1)
        else:
            o2 = Cropping2D(cropping=((0, 0) ,  (0, cx)))(o2)
        if (o1_height > o2_height):
            o1 = Cropping2D(cropping=((0, cy) ,  (0, 0)))(o1)
        else:
            o2 = Cropping2D(cropping=((0, cy) ,  (0, 0)))(o2)
        
        return o1, o2

    def batch_gen(self, path, batch_size):
        content_path, mask_path = get_file_paths(path)    
        content_path.sort()
        mask_path.sort()
        
        while True:
            index = random.sample(range(1, len(content_path)), batch_size)
            try:
                offset_h = random.randint(0, (2448-512))
                offset_w = random.randint(0, (2448-512))
                offset = (offset_h, offset_w)
                
                # contents = [vgg_sub_mean(get_image(content_path[i])) for i in index]
                # masks = [mask_preprocess(get_image(mask_path[i])) for i in index]
                contents = [vgg_sub_mean(random_crop(get_image(content_path[i]), offset)) for i in index]
                masks = [mask_preprocess(random_crop(get_image(mask_path[i]), offset)) for i in index]

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
                            # metrics=['accuracy'])
                            metrics=['accuracy', mean_IoU])
    
    def train(self, batch_size, epochs, steps, train_path, val_path, cp_cb):
        self.model.fit_generator(self.batch_gen(train_path, batch_size),
                                epochs=epochs,
                                steps_per_epoch=steps,
                                validation_data=self.batch_gen(val_path, batch_size),
                                validation_steps=50,
                                verbose=1,
                                callbacks=[cp_cb])

    # def load(self, path):
    #     self.model.load_weights(path, by_name=True)
    
    def decode(self, x):
        return self.model.predict(x)

    def summary(self):
        self.model.summary()
    
    def plot_model(self):
        plot_model(self.model, tofile='Vgg16_FCN.png')

