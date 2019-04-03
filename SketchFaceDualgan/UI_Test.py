from __future__ import print_function, division

import generate_dataset
import load_dataset
from keras.preprocessing import image
from keras.models import model_from_json
from keras.backend import manual_variable_initialization



class GAN:
    def __init__(self):

        self.sketch_rows = 128
        self.sketch_cols = 124
        self.channels = 1
        self.sketch_shape = (self.sketch_rows, self.sketch_cols, self.channels)
        self.sketch_size = self.sketch_rows * self.sketch_cols * self.channels

        # manual_variable_initialization(True)

        with open('models/generator_architecture.json', 'r') as f:
            self.generator = model_from_json(f.read())
        self.generator.load_weights('models/generator_weights.h5')
        # manual_variable_initialization(True)

        # self.generator = load_model('models/generator.h5')


    def run(self):
        generate_dataset.build("UI/Input", self.sketch_cols, self.sketch_rows, 1)

        (x_train_sketch, _) = load_dataset.get("UI/Input")

        sketches = x_train_sketch.reshape(len(x_train_sketch), self.sketch_rows, self.sketch_cols, self.channels)

        img_out = self.generator.predict(sketches)

        for i in range(0, len(img_out)):
            img = image.array_to_img(img_out[i])
            img.save('UI/Output/' + str(i) + '.png')


if __name__ == '__main__':
    gan = GAN()
    gan.run()
