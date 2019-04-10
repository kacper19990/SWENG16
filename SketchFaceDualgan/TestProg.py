from __future__ import print_function, division

import generate_dataset
import load_dataset
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.preprocessing import image
from keras.optimizers import Adam
import keras.backend as K
from keras.backend import manual_variable_initialization
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

        self.sketch_rows = 64
        self.sketch_cols = 62
        self.sketch_shape = (self.sketch_rows, self.sketch_cols, self.channels)
        self.sketch_size = self.sketch_rows * self.sketch_cols * self.channels

        optimizer = Adam(0.0002, 0.5)



        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=self.sketch_shape)
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

        # self.generator = self.build_generator()
        # self.generator.compile(loss='binary_crossentropy',
        #     optimizer=optimizer,
        #     metrics=['accuracy'])

        with open('models/generator_architecture.json', 'w') as f:
            f.write(self.generator.to_json())
        with open('models/discriminator_architecture.json', 'w') as f:
            f.write(self.discriminator.to_json())

        # manual_variable_initialization(True)

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

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
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))

        model.summary()

        noise = Input(shape=self.sketch_shape)
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
        generate_dataset.build("pictures", self.img_cols, self.img_rows, 0.7)

        # Load the dataset
        (x_train_pic, _) = load_dataset.get("pictures")

        generate_dataset.build("sketches", self.sketch_cols, self.sketch_rows, 0.7)

        (x_train_sketch, _) = load_dataset.get("sketches")

        tmp = list()

        for i in range(0, len(x_train_pic)):
            if (x_train_pic[i] is not None):
                tmp.append(x_train_pic[i])

        x_train_pic = np.asarray(tmp)


        # Rescale -1 to 1
        x_train_pic = x_train_pic / 127.5 - 1.
        x_train_pic = np.expand_dims(x_train_pic, axis=3)

        # if x_train_sketch is None:
        #     print("None\n")
        # else:
        #     print("Not None\n")

        x_train_sketch_ls = [x for x in x_train_sketch if x is not None]

        x_train_sketch = np.asarray(x_train_sketch_ls)

        x_train_sketch = x_train_sketch / 127.5 - 1.
        # x_train_sketch = self.crop_sketches(x_train_sketch)
        # x_train_sketch = np.expand_dims(x_train_sketch, axis=3)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, x_train_pic.shape[0], batch_size)
            imgs = x_train_pic[idx]

            idx = np.random.randint(0, x_train_sketch.shape[0], batch_size)
            sketches_list = x_train_sketch[idx]

            sketches = self.reshape_sketches(sketches_list)

            gen_imgs = self.generator.predict(sketches)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            idx = np.random.randint(0, x_train_sketch.shape[0], batch_size)
            sketches_list = x_train_sketch[idx]

            sketches = self.reshape_sketches(sketches_list)


            # noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Train the generator (to have the discriminator label samples as valid)
            g_loss = self.combined.train_on_batch(sketches, valid)
            # g_loss = self.generator.train_on_batch(sketches, imgs)
            # g_loss = self.combined.train_on_batch([sketches], [valid, imgs].asArray())

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch, x_train_sketch)
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
        if not os.path.exists('output_full/' + str(epoch)):
            os.makedirs('output_full/' + str(epoch))
        if not os.path.exists('output_full/' + str(epoch) + '/input'):
            os.makedirs('output_full/' + str(epoch) + '/input')
        if not os.path.exists('output_full/' + str(epoch) + '/output'):
            os.makedirs('output_full/' + str(epoch) + '/output')

        for i in range(batch_size):
            img = image.array_to_img(gen_imgs[i])
            img.save('output_full/' + str(epoch) + '/output/' + str(i) + '.png')
            img = image.array_to_img(sketches[i])
            img.save('output_full/' + str(epoch) + '/input/' + str(i) + '.png')


        #
        # fig, axs = plt.subplots(r, c)
        # cnt = 0
        # for i in range(r):
        #     for j in range(c):
        #         axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
        #         axs[i,j].axis('off')
        #         cnt += 1
        # fig.savefig("output/%d.png" % epoch)
        # plt.close()

    def reshape_sketches(self, sketches_list):
        sketches = np.asarray(sketches_list).reshape(len(sketches_list), self.sketch_rows, self.sketch_cols, self.channels)

        # Reshape sketches to size of smallest sketch
        # for i in range(0, len(sketches_list)):
        #     tmp = sketches_list[i]
        #     tmp = np.asarray(tmp).reshape(self.sketch_shape)
        #
        #     sketches[i] = tmp

        # sketches.reshape(len(sketches_list), self.sketch_rows, self.sketch_cols, self.channels)
        return sketches


if __name__ == '__main__':
    gan = GAN()
    if not os.path.exists('output_full'):
        os.makedirs('output_full')
    gan.train(epochs=100000, batch_size=64, sample_interval=200)
