from __future__ import print_function, division

import generate_dataset
import load_dataset
import dataset
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Lambda, Concatenate, multiply
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, Embedding
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.preprocessing import image
from keras.optimizers import Adam
import keras.backend as K
from keras.backend import manual_variable_initialization
import tensorflow as tf
import os


import matplotlib.pyplot as plt

import sys

import numpy as np


class GAN:
    def __init__(self):
        self.img_rows = 96
        self.img_cols = 64
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.img_size = self.img_rows * self.img_cols * self.channels

        self.sketch_rows = 96
        self.sketch_cols = 64
        self.sketch_shape = (self.sketch_rows, self.sketch_cols, self.channels)
        self.sketch_size = self.sketch_rows * self.sketch_cols * self.channels
        self.second_input_shape = (2, self.sketch_rows, self.sketch_cols, self.channels)

        optimizer = Adam(0.0001, 0.5)
        # optimizer_2 = Adam(0.0001, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=['binary_crossentropy'],
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label
        # noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(self.sketch_shape))
        img = self.generator([label])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated image as input and determines validity
        # and the label of that image
        valid = self.discriminator([img, label])

        # The combined model  (stacked generator and discriminator)
        # Trains generator to fool discriminator
        self.combined = Model(label, valid)
        self.combined.compile(loss=['binary_crossentropy'],
                              optimizer=optimizer)

        with open('models/generator_architecture.json', 'w') as f:
            f.write(self.generator.to_json())
        with open('models/discriminator_architecture.json', 'w') as f:
            f.write(self.discriminator.to_json())


    def build_generator(self):

        model = Sequential()

        model.add(Flatten(input_shape=self.sketch_shape))
        # model.add(Dense(64, input_dim=self.sketch_size))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(128, input_dim=self.sketch_size))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        # model.add(Dense(1024))
        # model.add(LeakyReLU(alpha=0.2))
        # model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))

        model.summary()

        noise = Input(shape=self.sketch_shape)
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Dense(512, input_dim=np.prod(self.img_shape)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=self.img_shape)
        label = Input(shape=self.sketch_shape, dtype='int32')
        # label = Input(shape=self.sketch_shape)

        # label_embedding = Flatten()(Embedding(self.img_size, np.prod(self.img_shape))(label))
        label_embedding = Flatten()(Embedding(self.img_size, 1)(label))
        # label_embedding = Flatten()(Embedding(256, np.prod(self.img_shape))(label))
        flat_img = Flatten()(img)
        # flat_label = Flatten()(label)
        # flat_label = Reshape([1, self.img_size])(flat_label)

        model_input = multiply([flat_img, label_embedding])

        validity = model(model_input)

        return Model([img, label], validity)

    def train(self, epochs, batch_size=128, sample_interval=50):

        # Load the dataset
        # # Generate the dataset
        dataset.generate("sketches", self.img_cols, self.img_rows, 0.95)
        dataset.generate("pictures", self.img_cols, self.img_rows, 0.95)

        # Load the dataset
        (X_train, X_test) = dataset.load("sketches")
        (Y_train, Y_test) = dataset.load("pictures")

        # Rescale -1 to 1
        X_train = X_train / 127.5 - 1.
        X_train = np.expand_dims(X_train, axis=3)

        X_test = X_test / 127.5 - 1.
        X_test = np.expand_dims(X_test, axis=3)

        y_train = Y_train / 127.5 - 1.
        y_train = np.expand_dims(Y_train, axis=3)

        Y_test = Y_test / 127.5 - 1.
        Y_test = np.expand_dims(Y_test, axis=3)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs, sketches = X_train[idx], y_train[idx]

            # Sample noise as generator input
            # noise = np.random.normal(0, 1, (batch_size, 100))

            # Generate a half batch of new images
            gen_imgs = self.generator.predict([sketches])

            # sketches = np.asarray(sketches).reshape(batch_size, self.img_size)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch([imgs, sketches], valid)
            d_loss_fake = self.discriminator.train_on_batch([gen_imgs, sketches], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Condition on sketches
            # sampled_sketches = np.random.randint(0, 10, batch_size).reshape(-1, 1)

            # Train the generator
            g_loss = self.combined.train_on_batch([sketches], valid)

            # Plot the progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch, X_train)


            # Plot the progress
            # print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
            # print ("%d [D2 loss: %f, acc.: %.2f%%] [G2 loss: %f]" % (epoch, d2_loss[0], 100*d2_loss[1], g2_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch, X_train)
                self.generator.save('generator.h5')
                self.discriminator.save('discriminator.h5')
                self.generator.save_weights('models/generator_weights.h5')
                self.discriminator.save_weights('models/discriminator_weights.h5')


    def sample_images(self, epoch, x_train_sketch):
        r, c = 5, 5
        # noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        batch_size = 16
        idx = np.random.randint(0, x_train_sketch.shape[0], batch_size)
        sketches_list = x_train_sketch[idx]

        sketches = self.reshape_sketches(sketches_list)
        # sketches = np.asarray(sketches_list)
        gen_imgs = self.generator.predict(sketches)
        if not os.path.exists('cgan_out/' + str(epoch)):
            os.makedirs('cgan_out/' + str(epoch))
        if not os.path.exists('cgan_out/' + str(epoch) + '/input'):
            os.makedirs('cgan_out/' + str(epoch) + '/input')
        if not os.path.exists('cgan_out/' + str(epoch) + '/output'):
            os.makedirs('cgan_out/' + str(epoch) + '/output')

        for i in range(batch_size):
            img = image.array_to_img(gen_imgs[i])
            img.save('cgan_out/' + str(epoch) + '/output/' + str(i) + '.png')
            img = image.array_to_img(sketches[i])
            img.save('cgan_out/' + str(epoch) + '/input/' + str(i) + '.png')

    def reshape_sketches(self, sketches_list):
        sketches = np.asarray(sketches_list).reshape(len(sketches_list), self.sketch_rows, self.sketch_cols, self.channels)
        return sketches


if __name__ == '__main__':
    gan = GAN()
    if not os.path.exists('2_discrim_out'):
        os.makedirs('2_discrim_out')
    gan.train(epochs=200000, batch_size=32, sample_interval=200)
