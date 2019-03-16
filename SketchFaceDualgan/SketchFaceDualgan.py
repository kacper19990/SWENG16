# Imports
from keras.layers import Input, Dense
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam

# Dualgan class
class Dualgan():

    # Constructor
    def __init__(self):
        # TODO: 1)modify dims appropriately (96*64)? 2)Add channels?
        self.image_rows = 28
        self.image_columns = 28
        # self.channels
        self.image_size = self.image_rows * self.image_columns

        # Build discriminator
        self.discriminator = self.build_discriminator()


    # Builds up the discriminator model
    def build_discriminator(self):

        # instantiate Input tensor with expected image dimensions
        image = Input(shape = (self.image_size,))

        # TODO: modify model structure?
        # instantiate model with hidden layers
        model = Sequential() # => we're using multiple layers one after another
        model.add(Dense(512, input_dim = self.image_size)) # ouput dim = 512
        model.add(LeakyReLU(alpha = 0.1)) # LeakyReLu > flexibility than ReLu
        model.add(Dense(256)) # ouput dim = 256 (in_dim = out_dim of prev layer)
        model.add(LeakyReLU(alpha = 0.1))
        model.add(BatchNormalization(momentum = 0.8)) # normalise activations of previous layer (mean -> 0, standard dev -> 1)
        model.add(Dense(1)) # ouput dim = 1

        authenticity = model(image) # Tensor with values of model's results of image authenticity

        return Model(inputs = image, outputs = authenticity) # discriminator with input as image and output as authenticity results
