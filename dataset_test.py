import generate_dataset
import load_dataset
from matplotlib import pyplot
from scipy.misc import toimage


# Show 16 images on a grid
def show_imgs(X):
    pyplot.figure(1)
    k = 0
    for i in range(0, 4):
        for j in range(0, 4):
            pyplot.subplot2grid((4, 4), (i, j))
            pyplot.imshow(toimage(X[k]))
            k = k + 1
    pyplot.show()


# Generate a zip file containing train and test numpy arrays
generate_dataset.build("cropped_sketch", 0.9)

# Extract train and test numpy arrays
x_train, x_test = load_dataset.get("cropped_sketch")

print(x_train.shape)
print(x_test.shape)

show_imgs(x_test[:16])
