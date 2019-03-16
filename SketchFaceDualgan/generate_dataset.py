import numpy as np
import os
import cv2


def build(dataset, width, height, train_portion):
    # Get path to dataset
    root = os.path.dirname(os.path.abspath(__file__))
    path = root + "/" + dataset + "/"
    directory = os.listdir(path)

    # Get filenames from directory and sort
    files = []
    for file in directory:
        files.append(file)
    files.sort()

    # Resize images and add to list
    images = []
    for file in files:
        image = cv2.imread(path + file, 0)
        image = cv2.resize(image, (width, height))
        images.append(image)

    # Divide into training and test segments
    split_point = int(len(images) * train_portion)

    # Put images into numpy arrays
    x_train = np.array(images[:split_point])
    x_test = np.array(images[split_point:])

    # Save arrays as zip file
    np.savez(dataset, x_train, x_test)
