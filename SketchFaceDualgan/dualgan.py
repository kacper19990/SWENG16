from __future__ import print_function, division

from keras.preprocessing import image
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import RMSprop, Adam
from keras.utils import to_categorical
import keras.backend as K

import numpy as np
import os
import dataset


class DUALGAN:
    def __init__(self):
        self.img_rows = 96
        self.img_cols = 64
        self.channels = 1
        self.img_dim = self.img_rows * self.img_cols

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminators
        self.D_A = self.build_discriminator()
        self.D_A.compile(loss=self.wasserstein_loss,
            optimizer=optimizer,
            metrics=['accuracy'])
        self.D_B = self.build_discriminator()
        self.D_B.compile(loss=self.wasserstein_loss,
            optimizer=optimizer,
            metrics=['accuracy'])

        # -------------------------
        # Construct Computational
        #   Graph of Generators
        # -------------------------

        # Build the generators
        self.G_AB = self.build_generator()
        self.G_BA = self.build_generator()

        # For the combined model we will only train the generators
        self.D_A.trainable = False
        self.D_B.trainable = False

        # The generator takes images from their respective domains as inputs
        imgs_A = Input(shape=(self.img_dim,))
        imgs_B = Input(shape=(self.img_dim,))

        # Generators translates the images to the opposite domain
        fake_B = self.G_AB(imgs_A)
        fake_A = self.G_BA(imgs_B)

        # The discriminators determines validity of translated images
        valid_A = self.D_A(fake_A)
        valid_B = self.D_B(fake_B)

        # Generators translate the images back to their original domain
        recov_A = self.G_BA(fake_B)
        recov_B = self.G_AB(fake_A)

        # The combined model  (stacked generators and discriminators)
        self.combined = Model(inputs=[imgs_A, imgs_B], outputs=[valid_A, valid_B, recov_A, recov_B])
        self.combined.compile(loss=[self.wasserstein_loss, self.wasserstein_loss, 'mae', 'mae'],
                            optimizer=optimizer,
                            loss_weights=[1, 1, 100, 100])

    def build_generator(self):

        X = Input(shape=(self.img_dim,))

        model = Sequential()
        model.add(Dense(256, input_dim=self.img_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dropout(0.4))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dropout(0.4))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dropout(0.4))
        model.add(Dense(self.img_dim, activation='tanh'))

        X_translated = model(X)

        return Model(X, X_translated)

    def build_discriminator(self):

        img = Input(shape=(self.img_dim,))

        model = Sequential()
        model.add(Dense(512, input_dim=self.img_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1))

        validity = model(img)

        return Model(img, validity)

    def sample_generator_input(self, X, batch_size):
        # Sample random batch of images from X
        idx = np.random.randint(0, X.shape[0], batch_size)
        return X[idx]

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def train(self, epochs, batch_size=128, sample_interval=50):
        # Generate the dataset
        dataset.generate("sketches", self.img_cols, self.img_rows, 0.95)
        dataset.generate("pictures", self.img_cols, self.img_rows, 0.95)

        # Load the dataset
        (A_train, A_test) = dataset.load("sketches")
        (B_train, B_test) = dataset.load("pictures")

        # Rescale -1 to 1
        A_train = (A_train.astype(np.float32) - 127.5) / 127.5
        B_train = (B_train.astype(np.float32) - 127.5) / 127.5
        A_test = (A_test.astype(np.float32) - 127.5) / 127.5
        B_test = (B_test.astype(np.float32) - 127.5) / 127.5

        # Reshape for generators / discriminators
        A_train = A_train.reshape(A_train.shape[0], self.img_dim)
        B_train = B_train.reshape(B_train.shape[0], self.img_dim)
        A_test = A_test.reshape(A_test.shape[0], self.img_dim)
        B_test = B_test.reshape(B_test.shape[0], self.img_dim)

        clip_value = 0.01
        n_critic = 4

        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake = np.ones((batch_size, 1))

        for epoch in range(epochs):

            # Train the discriminator for n_critic iterations
            for _ in range(n_critic):

                # ----------------------
                #  Train Discriminators
                # ----------------------

                # Sample generator inputs
                imgs_A = self.sample_generator_input(A_train, batch_size)
                imgs_B = self.sample_generator_input(B_train, batch_size)

                # Translate images to their opposite domain
                fake_B = self.G_AB.predict(imgs_A)
                fake_A = self.G_BA.predict(imgs_B)

                # Train the discriminators
                D_A_loss_real = self.D_A.train_on_batch(imgs_A, valid)
                D_A_loss_fake = self.D_A.train_on_batch(fake_A, fake)

                D_B_loss_real = self.D_B.train_on_batch(imgs_B, valid)
                D_B_loss_fake = self.D_B.train_on_batch(fake_B, fake)

                D_A_loss = 0.5 * np.add(D_A_loss_real, D_A_loss_fake)
                D_B_loss = 0.5 * np.add(D_B_loss_real, D_B_loss_fake)

                # Clip discriminator weights
                for d in [self.D_A, self.D_B]:
                    for l in d.layers:
                        weights = l.get_weights()
                        weights = [np.clip(w, -clip_value, clip_value) for w in weights]
                        l.set_weights(weights)

            # ------------------
            #  Train Generators
            # ------------------

            # Train the generators
            g_loss = self.combined.train_on_batch([imgs_A, imgs_B], [valid, valid, imgs_A, imgs_B])

            # Plot the progress
            print ("%d [D1 loss: %f] [D2 loss: %f] [G loss: %f]" \
                % (epoch, D_A_loss[0], D_B_loss[0], g_loss[0]))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.save_imgs(epoch, A_test, B_test)
                self.save_models(epoch)

    def save_imgs(self, epoch, A_test, B_test):
        sample_size = 32

        # Sample generator inputs
        # imgs_A = self.sample_generator_input(A_test, sample_size)
        # imgs_B = self.sample_generator_input(B_test, sample_size)

        imgs_A = A_test[0:sample_size]
        imgs_B = B_test[0:sample_size]

        # Images translated to their opposite domain
        fake_B = self.G_AB.predict(imgs_A)
        fake_A = self.G_BA.predict(imgs_B)

        # Rescale images 0 - 1
        imgs_A = 0.5 * imgs_A + 0.5
        imgs_B = 0.5 * imgs_B + 0.5
        fake_A = 0.5 * fake_A + 0.5
        fake_B = 0.5 * fake_B + 0.5

        # Reshape images
        imgs_A = imgs_A.reshape((sample_size, self.img_rows, self.img_cols, self.channels))
        imgs_B = imgs_B.reshape((sample_size, self.img_rows, self.img_cols, self.channels))
        fake_A = fake_A.reshape((sample_size, self.img_rows, self.img_cols, self.channels))
        fake_B = fake_B.reshape((sample_size, self.img_rows, self.img_cols, self.channels))

        # Make directories
        if not os.path.exists('dualgan_results/sketch2face/sketches/' + str(epoch)):
            os.makedirs('dualgan_results/sketch2face/sketches/' + str(epoch))
            
        if not os.path.exists('dualgan_results/sketch2face/pictures/' + str(epoch)):
            os.makedirs('dualgan_results/sketch2face/pictures/' + str(epoch))
            
        if not os.path.exists('dualgan_results/sketch2face/output/' + str(epoch)):
            os.makedirs('dualgan_results/sketch2face/output/' + str(epoch))
            
        if not os.path.exists('dualgan_results/face2sketch/pictures/' + str(epoch)):
            os.makedirs('dualgan_results/face2sketch/pictures/' + str(epoch))

        if not os.path.exists('dualgan_results/face2sketch/sketches/' + str(epoch)):
            os.makedirs('dualgan_results/face2sketch/sketches/' + str(epoch))

        if not os.path.exists('dualgan_results/face2sketch/output/' + str(epoch)):
            os.makedirs('dualgan_results/face2sketch/output/' + str(epoch))

        # Save images
        for i in range(len(imgs_A)):
            img = image.array_to_img(imgs_A[i])
            img.save('dualgan_results/sketch2face/sketches/' + str(epoch) + '/' + str(i) + '.png')

        for i in range(len(imgs_B)):
            img = image.array_to_img(imgs_B[i])
            img.save('dualgan_results/sketch2face/pictures/' + str(epoch) + '/' + str(i) + '.png')

        for i in range(len(fake_B)):
            img = image.array_to_img(fake_B[i])
            img.save('dualgan_results/sketch2face/output/' + str(epoch) + '/' + str(i) + '.png')

        for i in range(len(imgs_B)):
            img = image.array_to_img(imgs_B[i])
            img.save('dualgan_results/face2sketch/pictures/' + str(epoch) + '/' + str(i) + '.png')
            
        for i in range(len(imgs_A)):
            img = image.array_to_img(imgs_A[i])
            img.save('dualgan_results/face2sketch/sketches/' + str(epoch) + '/' + str(i) + '.png')

        for i in range(len(fake_A)):
            img = image.array_to_img(fake_A[i])
            img.save('dualgan_results/face2sketch/output/' + str(epoch) + '/' + str(i) + '.png')

    def save_models(self, epoch):
        # Make directories
        if not os.path.exists('dualgan_results/sketch2face/models/' + str(epoch)):
            os.makedirs('dualgan_results/sketch2face/models/' + str(epoch))

        if not os.path.exists('dualgan_results/face2sketch/models/' + str(epoch)):
            os.makedirs('dualgan_results/face2sketch/models/' + str(epoch))

        model = self.G_AB.to_json()
        with open("dualgan_results/sketch2face/models/" + str(epoch) + "/generator.json", "w") as json_file:
            json_file.write(model)
        self.G_AB.save_weights("dualgan_results/sketch2face/models/" + str(epoch) + "/generator.h5")

        model = self.D_A.to_json()
        with open("dualgan_results/sketch2face/models/" + str(epoch) + "/discriminator.json", "w") as json_file:
            json_file.write(model)
        self.D_A.save_weights("dualgan_results/sketch2face/models/" + str(epoch) + "/discriminator.h5")

        model = self.G_BA.to_json()
        with open("dualgan_results/face2sketch/models/" + str(epoch) + "/generator.json", "w") as json_file:
            json_file.write(model)
        self.G_BA.save_weights("dualgan_results/face2sketch/models/" + str(epoch) + "/generator.h5")

        model = self.D_B.to_json()
        with open("dualgan_results/face2sketch/models/" + str(epoch) + "/discriminator.json", "w") as json_file:
            json_file.write(model)
        self.D_B.save_weights("dualgan_results/face2sketch/models/" + str(epoch) + "/discriminator.h5")


def test():
    gan = DUALGAN()
    gan.train(epochs=100000, batch_size=32, sample_interval=1000)


if __name__ == '__main__':
    test()
