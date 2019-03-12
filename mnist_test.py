from matplotlib import pyplot
from PIL import Image
from keras.datasets import mnist


def show_images(x):
    pyplot.figure(1)
    k = 0
    for i in range(0, 4):
        for j in range(0, 4):
            pyplot.subplot2grid((4, 4), (i, j))
            pyplot.imshow(Image.fromarray(x[k], 'P'))
            k = k + 1
    pyplot.show()


(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

show_images(x_test[:16])
