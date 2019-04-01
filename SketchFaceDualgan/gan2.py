from __future__ import print_function, division

import dataset
import os

from keras.preprocessing import image
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt

import sys

import numpy as np

class GAN():
    def __init__(self):
        self.img_rows = 192
        self.img_cols = 128
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.img_size = self.img_rows * self.img_cols * self.channels

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.img_size,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)


    def build_generator(self):

        model = Sequential()

        model.add(Dense(256, input_dim=self.img_size))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))

        model.summary()

        noise = Input(shape=(self.img_size,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128, sample_interval=50):
        # Generate the dataset
        dataset.generate("sketches", self.img_cols, self.img_rows, 1)
        dataset.generate("pictures", self.img_cols, self.img_rows, 1)

        # Load the dataset
        (X_train, _) = dataset.load("sketches")
        (Y_train, _) = dataset.load("pictures")

        # Rescale -1 to 1
        X_train = X_train / 127.5 - 1.
        X_train = np.expand_dims(X_train, axis=3)

        Y_train = Y_train / 127.5 - 1.
        Y_train = np.expand_dims(Y_train, axis=3)

        X_train = X_train.reshape(X_train.shape[0], self.img_size)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            sketches = X_train[idx]
            pictures = Y_train[idx]

            # Generate a batch of new images
            gen_imgs = self.generator.predict(sketches)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(pictures, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            sketches = X_train[idx]

            # Train the generator (to have the discriminator label samples as valid)
            g_loss = self.combined.train_on_batch(sketches, valid)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch, X_train, Y_train)

    def sample_images(self, epoch, X_train, Y_train):
        sample_size = 5

        idx = np.random.randint(0, X_train.shape[0], sample_size)
        sketches = X_train[idx]
        pictures = Y_train[idx]

        gen_imgs = self.generator.predict(sketches)

        sketches = sketches.reshape(sample_size, self.img_rows, self.img_cols, self.channels)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        if not os.path.exists('gan_results'):
            os.makedirs('gan_results')

        if not os.path.exists('gan_results/pictures'):
            os.makedirs('gan_results/pictures')

        if not os.path.exists('gan_results/sketches'):
            os.makedirs('gan_results/sketches')

        if not os.path.exists('gan_results/output'):
            os.makedirs('gan_results/output')

        if not os.path.exists('gan_results/pictures/' + str(epoch)):
            os.makedirs('gan_results/pictures/' + str(epoch))

        if not os.path.exists('gan_results/sketches/' + str(epoch)):
            os.makedirs('gan_results/sketches/' + str(epoch))

        if not os.path.exists('gan_results/output/' + str(epoch)):
            os.makedirs('gan_results/output/' + str(epoch))

        for i in range(len(sketches)):
            img = image.array_to_img(sketches[i])
            img.save('gan_results/sketches/' + str(epoch) + '/' + str(i) + '.png')

        for i in range(len(pictures)):
            img = image.array_to_img(pictures[i])
            img.save('gan_results/pictures/' + str(epoch) + '/' + str(i) + '.png')

        for i in range(len(gen_imgs)):
            img = image.array_to_img(gen_imgs[i])
            img.save('gan_results/output/' + str(epoch) + '/' + str(i) + '.png')


if __name__ == '__main__':
    gan = GAN()
    gan.train(epochs=30000, batch_size=32, sample_interval=200)