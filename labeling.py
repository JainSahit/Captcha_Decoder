from sklearn.datasets import load_digits
import cv2

import numpy as np
digits = load_digits()
print(digits)
cv2.imshow("image", digits.images[-1])
cv2.waitKey(0)
