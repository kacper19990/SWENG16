from keras.preprocessing import image
from keras.models import model_from_json

import numpy as np
import os
import dataset


def generate_samples(model_path, img_size, sample_size):
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

    # Make directory
    if not os.path.exists("model_results"):
        os.makedirs("model_results")

    # Save images
    for i in range(len(gen_imgs)):
        img = image.array_to_img(gen_imgs[i])
        img.save('model_results/' + str(i) + '.png')


generate_samples("gan_results/models/100/generator", 192*128, 32)
