import dataset
from matplotlib import pyplot
from PIL import Image


# Show 16 images on a grid
def show_images(x):
    pyplot.figure(1)
    k = 0
    for i in range(0, 4):
        for j in range(0, 4):
            pyplot.subplot2grid((4, 4), (i, j))
            pyplot.imshow(Image.fromarray(x[k], 'P'))
            k = k + 1
    pyplot.show()


# Generate a zip file containing train and test numpy arrays
dataset.generate("sketches", 128, 192, 0.9)
dataset.generate("pictures", 128, 192, 0.9)

# Extract train and test numpy arrays
x_train, x_test = dataset.load("sketches")
y_train, y_test = dataset.load("pictures")

# Check array dimensions
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# Display datasets
show_images(x_train[:16])
show_images(x_test[:16])
show_images(y_train[:16])
show_images(y_test[:16])
