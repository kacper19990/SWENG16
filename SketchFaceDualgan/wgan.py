from __future__ import print_function, division

from keras.preprocessing import image
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import RMSprop

import keras.backend as K

import numpy as np
import dataset
import os


class WGAN:
    def __init__(self):
        self.img_rows = 192
        self.img_cols = 128
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.img_size = self.img_rows * self.img_cols * self.channels

        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 5
        self.clip_value = 0.01
        optimizer = RMSprop(lr=0.00005)

        # Build and compile the critic
        self.critic = self.build_critic()
        self.critic.compile(loss=self.wasserstein_loss,
                            optimizer=optimizer,
                            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes sketches as input and generated pictures
        z = Input(shape=(self.img_size,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.critic.trainable = False

        # The critic takes generated images as input and determines validity
        valid = self.critic(img)

        # The combined model  (stacked generator and critic)
        self.combined = Model(z, valid)
        self.combined.compile(loss=self.wasserstein_loss,
                              optimizer=optimizer,
                              metrics=['accuracy'])

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

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

    def build_critic(self):

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

            for _ in range(self.n_critic):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random batch of images
                idx = np.random.randint(0, X_train.shape[0], batch_size)
                sketches = X_train[idx]
                pictures = Y_train[idx]

                # Generate a batch of new images
                gen_imgs = self.generator.predict(sketches)

                # Train the critic
                d_loss_real = self.critic.train_on_batch(pictures, valid)
                d_loss_fake = self.critic.train_on_batch(gen_imgs, fake)
                d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

                # Clip critic weights
                for l in self.critic.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
                    l.set_weights(weights)

            # ---------------------
            #  Train Generator
            # ---------------------

            g_loss = self.combined.train_on_batch(sketches, valid)

            # Plot the progress
            print("%d [D loss: %f] [G loss: %f]" % (epoch, 1 - d_loss[0], 1 - g_loss[0]))

            # If at save interval => save generated image samples
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
        if not os.path.exists('wgan_results/pictures/' + str(epoch)):
            os.makedirs('wgan_results/pictures/' + str(epoch))

        if not os.path.exists('wgan_results/sketches/' + str(epoch)):
            os.makedirs('wgan_results/sketches/' + str(epoch))

        if not os.path.exists('wgan_results/output/' + str(epoch)):
            os.makedirs('wgan_results/output/' + str(epoch))

        # Save images
        for i in range(len(sketches)):
            img = image.array_to_img(sketches[i])
            img.save('wgan_results/sketches/' + str(epoch) + '/' + str(i) + '.png')

        for i in range(len(pictures)):
            img = image.array_to_img(pictures[i])
            img.save('wgan_results/pictures/' + str(epoch) + '/' + str(i) + '.png')

        for i in range(len(gen_imgs)):
            img = image.array_to_img(gen_imgs[i])
            img.save('wgan_results/output/' + str(epoch) + '/' + str(i) + '.png')

    def save_models(self, epoch):
        # Make directory
        if not os.path.exists('wgan_results/models/' + str(epoch)):
            os.makedirs('wgan_results/models/' + str(epoch))

        model = self.generator.to_json()
        with open("wgan_results/models/" + str(epoch) + "/generator.json", "w") as json_file:
            json_file.write(model)
        self.generator.save_weights("wgan_results/models/" + str(epoch) + "/generator.h5")

        model = self.critic.to_json()
        with open("wgan_results/models/" + str(epoch) + "/discriminator.json", "w") as json_file:
            json_file.write(model)
        self.critic.save_weights("wgan_results/models/" + str(epoch) + "/discriminator.h5")


if __name__ == '__main__':
    wgan = WGAN()
    wgan.train(epochs=100000, batch_size=32, sample_interval=100)
