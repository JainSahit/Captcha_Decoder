import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import string

symbols = string.ascii_lowercase + "0123456789"  # All symbols captcha can contain
num_symbols = len(symbols)
img_shape = (50, 200, 1)

path = "/Users/sahitjain/Desktop/Fall-19/IFT-598/Project/captcha_dataset/samples"


def preprocess_data():
    n_samples = len(os.listdir(path))
    X = np.zeros((n_samples, 50, 200, 1))
    y = np.zeros((5, n_samples, num_symbols))
    print(n_samples)
    for i, pic in enumerate(os.listdir(path)):
        # Read image as grayscale
        img = cv2.imread(os.path.join(path, pic), cv2.IMREAD_GRAYSCALE)


        pic_target = pic[:-4]
        print(pic_target)
        if len(pic_target) < 6:
            # Scale and reshape image
            img = img / 255.0
            img = np.reshape(img, (50, 200, 1))

            # Define targets and code them using OneHotEncoding
            targs = np.zeros((5, num_symbols))
            for j, l in enumerate(pic_target):
                ind = symbols.find(l)
                targs[j, ind] = 1
            X[i] = img
            y[:, i] = targs

    # Return final data
    print(len(img))
    return X, y


X, y = preprocess_data()
X_train, y_train = X[:970], y[:, :970]
X_test, y_test = X[970:], y[:, 970:]
