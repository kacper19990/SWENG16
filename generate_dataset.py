import numpy as np
import os
import cv2


def build(dataset, test_portion):
    root = os.path.dirname(os.path.abspath(__file__))
    path = root + "/" + dataset + "/"
    directory = os.listdir(path)

    images = []

    for file in directory:
        image = cv2.imread(path + file, 0)
        images.append(image)

    split_point = int(len(images) * test_portion)

    x_test = np.array(images[:split_point])
    x_train = np.array(images[split_point:])

    np.savez(dataset, x_test, x_train)
