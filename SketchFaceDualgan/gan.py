from __future__ import print_function, division

from keras.preprocessing import image
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import numpy as np
import dataset
import os


class GAN:
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

        # The generator takes sketches as input and generates pictures
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

        sketch = Input(shape=(self.img_size,))
        picture = model(sketch)

        return Model(sketch, picture)

    def build_discriminator(self):

        model = Sequential()

        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        picture = Input(shape=self.img_shape)
        validity = model(picture)

        return Model(picture, validity)

    def train(self, epochs, batch_size=128, sample_interval=50):
        # Generate the dataset
        dataset.generate("sketches", self.img_cols, self.img_rows, 0.9)
        dataset.generate("pictures", self.img_cols, self.img_rows, 0.9)

        # Load the dataset
        (X_train, X_test) = dataset.load("sketches")
        (Y_train, Y_test) = dataset.load("pictures")

        # Rescale -1 to 1
        X_train = X_train / 127.5 - 1.
        X_train = np.expand_dims(X_train, axis=3)

        X_test = X_test / 127.5 - 1.
        X_test = np.expand_dims(X_test, axis=3)

        Y_train = Y_train / 127.5 - 1.
        Y_train = np.expand_dims(Y_train, axis=3)

        Y_test = Y_test / 127.5 - 1.
        Y_test = np.expand_dims(Y_test, axis=3)

        # Reshape sketches for generator
        X_train = X_train.reshape(X_train.shape[0], self.img_size)
        X_test = X_test.reshape(X_test.shape[0], self.img_size)

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
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples and models
            if epoch % sample_interval == 0:
                self.sample_images(epoch, X_test, Y_test)
                self.save_models(epoch)

    def sample_images(self, epoch, X_test, Y_test):
        sample_size = 32

        # idx = np.random.randint(0, X_train.shape[0], sample_size)
        # sketches = X_test[idx]
        # pictures = Y_test[idx]

        sketches = X_test[0:sample_size]
        pictures = Y_test[0:sample_size]

        gen_imgs = self.generator.predict(sketches)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        sketches = sketches.reshape(sample_size, self.img_rows, self.img_cols, self.channels)

        # Make directories
        if not os.path.exists('gan_results/pictures/' + str(epoch)):
            os.makedirs('gan_results/pictures/' + str(epoch))

        if not os.path.exists('gan_results/sketches/' + str(epoch)):
            os.makedirs('gan_results/sketches/' + str(epoch))

        if not os.path.exists('gan_results/output/' + str(epoch)):
            os.makedirs('gan_results/output/' + str(epoch))

        # Save images
        for i in range(len(sketches)):
            img = image.array_to_img(sketches[i])
            img.save('gan_results/sketches/' + str(epoch) + '/' + str(i) + '.png')

        for i in range(len(pictures)):
            img = image.array_to_img(pictures[i])
            img.save('gan_results/pictures/' + str(epoch) + '/' + str(i) + '.png')

        for i in range(len(gen_imgs)):
            img = image.array_to_img(gen_imgs[i])
            img.save('gan_results/output/' + str(epoch) + '/' + str(i) + '.png')

    def save_models(self, epoch):
        # Make directory
        if not os.path.exists('gan_results/models/' + str(epoch)):
            os.makedirs('gan_results/models/' + str(epoch))

        generator_model = self.generator.to_json()
        with open("gan_results/models/" + str(epoch) + "/generator.json", "w") as json_file:
            json_file.write(generator_model)
        self.generator.save_weights("gan_results/models/" + str(epoch) + "/generator.h5")

        discriminator_model = self.discriminator.to_json()
        with open("gan_results/models/" + str(epoch) + "/discriminator.json", "w") as json_file:
            json_file.write(discriminator_model)
        self.generator.save_weights("gan_results/models/" + str(epoch) + "/discriminator.h5")


if __name__ == '__main__':
    gan = GAN()
    gan.train(epochs=100000, batch_size=32, sample_interval=1000)
