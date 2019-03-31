import numpy as np
import os
import cv2


def generate(dataset, width, height, train_portion):
    # Get filenames from directory and sort
    directory = os.listdir(dataset)
    files = []
    for file in directory:
        files.append(file)
    files.sort()

    # Resize images and add to list
    images = []
    for file in files:
        image = cv2.imread(dataset + "/" + file, 0)
        image = cv2.resize(image, (width, height))
        images.append(image)

    # Divide into training and test segments
    split_point = int(len(images) * train_portion)

    # Put images into numpy arrays
    x_train = np.array(images[:split_point])
    x_test = np.array(images[split_point:])

    # Save arrays as zip file
    np.savez(dataset, x_train, x_test)


def load(dataset):
    # Load arrays from file
    f = np.load(dataset + ".npz")
    x_train = f['arr_0.npy']
    x_test = f['arr_1.npy']
    f.close()

    return x_train, x_test
