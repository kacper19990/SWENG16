from __future__ import print_function, division
import scipy

from keras.datasets import mnist
from normalization.instancenormalization import InstanceNormalization
import dataset
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.preprocessing import image
import datetime
import matplotlib.pyplot as plt
import sys
from data_loader import DataLoader
import numpy as np
import os

class Pix2Pix():
    def __init__(self):
        # Input shape
        self.img_rows = 256
        self.img_cols = 256
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        # # Configure data loader
        # self.dataset_name = 'facades'
        # self.data_loader = DataLoader(dataset_name=self.dataset_name,
        #                               img_res=(self.img_rows, self.img_cols))
        #

        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2**4)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 64
        self.df = 64

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])

        #-------------------------
        # Construct Computational
        #   Graph of Generator
        #-------------------------

        # Build the generator
        self.generator = self.build_generator()

        # Input images and their conditioning images
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # By conditioning on B generate a fake version of A
        fake_A = self.generator(img_B)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # Discriminators determines validity of translated images / condition pairs
        valid = self.discriminator([fake_A, img_B])

        self.combined = Model(inputs=[img_A, img_B], outputs=[valid, fake_A])
        self.combined.compile(loss=['mse', 'mae'],
                              loss_weights=[1, 100],
                              optimizer=optimizer)

    def build_generator(self):
        """U-Net Generator"""

        def conv2d(layer_input, filters, f_size=4, bn=True):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = BatchNormalization(momentum=0.8)(u)
            u = Concatenate()([u, skip_input])
            return u

        # Image input
        d0 = Input(shape=self.img_shape)

        # Downsampling
        d1 = conv2d(d0, self.gf, bn=False)
        d2 = conv2d(d1, self.gf*2)
        d3 = conv2d(d2, self.gf*4)
        d4 = conv2d(d3, self.gf*8)
        d5 = conv2d(d4, self.gf*8)
        d6 = conv2d(d5, self.gf*8)
        d7 = conv2d(d6, self.gf*8)

        # Upsampling
        u1 = deconv2d(d7, d6, self.gf*8)
        u2 = deconv2d(u1, d5, self.gf*8)
        u3 = deconv2d(u2, d4, self.gf*8)
        u4 = deconv2d(u3, d3, self.gf*4)
        u5 = deconv2d(u4, d2, self.gf*2)
        u6 = deconv2d(u5, d1, self.gf)

        u7 = UpSampling2D(size=2)(u6)
        # output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u7)
        output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='relu')(u7)

        return Model(d0, output_img)

    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, bn=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # Concatenate image and conditioning image by channels to produce input
        combined_imgs = Concatenate(axis=-1)([img_A, img_B])

        d1 = d_layer(combined_imgs, self.df, bn=False)
        d2 = d_layer(d1, self.df*2)
        d3 = d_layer(d2, self.df*4)
        d4 = d_layer(d3, self.df*8)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        return Model([img_A, img_B], validity)

    def train(self, epochs, batch_size=1, sample_interval=50):

        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        # dataset.generate("sketches", self.img_cols, self.img_rows, 0.95)
        # dataset.generate("pictures", self.img_cols, self.img_rows, 0.95)
        dataset.generate("100_sketches", self.img_cols, self.img_rows, 1)
        dataset.generate("100_pictures", self.img_cols, self.img_rows, 1)

        # Load the dataset
        # (X_train, X_test) = dataset.load("sketches")
        # (Y_train, Y_test) = dataset.load("pictures")
        (X_train, X_test) = dataset.load("100_sketches")
        (Y_train, Y_test) = dataset.load("100_pictures")

        # Rescale -1 to 1
        X_train = X_train / 127.5 - 1.
        X_train = np.expand_dims(X_train, axis=3)

        X_test = X_test / 127.5 - 1.
        X_test = np.expand_dims(X_test, axis=3)

        y_train = Y_train / 127.5 - 1.
        y_train = np.expand_dims(Y_train, axis=3)

        Y_test = Y_test / 127.5 - 1.
        Y_test = np.expand_dims(Y_test, axis=3)

        for epoch in range(epochs):
            # for batch_i, (imgs_A, imgs_B) in enumerate(self.data_loader.load_batch(batch_size)):
            idx = np.random.randint(0, Y_train.shape[0], batch_size)
            imgs_B = X_train[idx]
            imgs_A = Y_train[idx]
            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Condition on B and generate a translated version
            fake_A = self.generator.predict(imgs_B)

            imgs_A = np.reshape(imgs_A, [np.shape(imgs_A)[0], self.img_rows, self.img_cols, self.channels])
            # Train the discriminators (original images = real / generated = Fake)
            d_loss_real = self.discriminator.train_on_batch([imgs_A, imgs_B], valid)
            d_loss_fake = self.discriminator.train_on_batch([fake_A, imgs_B], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # -----------------
            #  Train Generator
            # -----------------

            # Train the generators
            g_loss = self.combined.train_on_batch([imgs_A, imgs_B], [valid, imgs_A])

            elapsed_time = datetime.datetime.now() - start_time
            # Plot the progress
            print ("[Epoch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f] time: %s" % (epoch, epochs,
                                                                    d_loss[0], 100*d_loss[1],
                                                                    g_loss[0],
                                                                    elapsed_time))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch, X_train)


    def sample_images(self, epoch, x_train_sketch):
        r, c = 5, 5
        # noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        batch_size = 16
        idx = np.random.randint(0, x_train_sketch.shape[0], batch_size)
        sketches_list = x_train_sketch[idx]

        sketches = np.asarray(sketches_list).reshape(len(sketches_list), self.img_rows, self.img_cols, self.channels)
        # sketches = np.asarray(sketches_list)
        gen_imgs = self.generator.predict(sketches)


        if not os.path.exists('p2p/' + str(epoch)):
            os.makedirs('p2p/' + str(epoch))
        if not os.path.exists('p2p/' + str(epoch) + '/input'):
            os.makedirs('p2p/' + str(epoch) + '/input')
        if not os.path.exists('p2p/' + str(epoch) + '/output'):
            os.makedirs('p2p/' + str(epoch) + '/output')

        for i in range(batch_size):
            img = image.array_to_img(gen_imgs[i])
            img.save('p2p/' + str(epoch) + '/output/' + str(i) + '.png')
            img = image.array_to_img(sketches[i])
            img.save('p2p/' + str(epoch) + '/input/' + str(i) + '.png')


if __name__ == '__main__':
    gan = Pix2Pix()
    gan.train(epochs=20000, batch_size=4, sample_interval=5)
