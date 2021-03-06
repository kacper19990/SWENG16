from keras.preprocessing import image
from keras.models import model_from_json

import numpy as np
import os
import dataset


def generate_samples(model_path, rows, cols, channels, sample_size):
    img_size = rows * cols

    # Load model
    json_file = open(model_path + ".json", 'r')
    json_model = json_file.read()
    json_file.close()
    model = model_from_json(json_model)
    model.load_weights(model_path + ".h5")

    # Load dataset
    (_, sketches) = dataset.load("sketches")
    sketches = sketches[0:sample_size]

    # Resize dataset
    sketches = sketches / 127.5 - 1.
    sketches = np.expand_dims(sketches, axis=3)
    sketches = sketches.reshape(sketches.shape[0], img_size)

    # Generate samples
    gen_imgs = model.predict(sketches)

    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5

    # Reshape images
    sketches = sketches.reshape((sample_size, rows, cols, channels))
    gen_imgs = gen_imgs.reshape((sample_size, rows, cols, channels))

    # Make directories
    if not os.path.exists("model_results/input"):
        os.makedirs("model_results/input")

    if not os.path.exists("model_results/output"):
        os.makedirs("model_results/output")

    # Save images
    for i in range(len(sketches)):
        img = image.array_to_img(sketches[i])
        img.save('model_results/input/' + str(i) + '.png')
        
    for i in range(len(gen_imgs)):
        img = image.array_to_img(gen_imgs[i])
        img.save('model_results/output/' + str(i) + '.png')


generate_samples("gan_results/models/100/generator", 192, 128, 1, 32)
