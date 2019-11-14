import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
kernel = np.ones((3,2),np.uint8)
#np.set_printoptions(threshold=sys.maxsize)

img = cv2.imread("/Users/sahitjain/Desktop/Fall-19/IFT-598/Project/captcha_dataset/samples/2cegf.png", 0)
#img = img[10:, 28:145]
print(img)
nIMG = np.array(img)
print(nIMG.shape)
inverted_img = (255.0 - img)
new_img = inverted_img / 255.0
print(new_img)

ret,thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
ret,thresh2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
ret,thresh3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
ret,thresh4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
ret,thresh6 = cv2.threshold(img, 127, 255, cv2.THRESH_OTSU)
dilate1 = cv2.dilate(thresh6, kernel, iterations=1)
errosion1 = cv2.erode(dilate1, kernel, iterations=1)
opening = cv2.morphologyEx(errosion1, cv2.MORPH_OPEN, kernel)
titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','OTSU','MORPH OPENING', 'DILATE', 'ERODE']
images = [img, thresh1, thresh2, thresh3, thresh4, thresh6, opening, dilate1, errosion1]

for i in range(9):
    plt.subplot(3,3,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()

x, y, w, h = 30, 12, 21, 38
for  i in range(5):
    #get the bounding rect
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.rectangle(thresh6, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.rectangle(opening, (x, y), (x + w, y + h), (0, 255, 0), 2)
    x += w

titles3 = ['Original', "OTSU", 'MORPH OPENING']
images3 = [img, thresh6, opening] #img, opening, opening2, opening3]

for i in range(3):
    plt.subplot(3, 1, i + 1), plt.imshow(images3[i], 'gray')
    plt.title(titles3[i])
    plt.xticks([]), plt.yticks([])

plt.title('Contouring')
plt.show()

plt.hist(img.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')
histr = cv2.calcHist([img],[0],None,[256],[0,256])
plt.plot(histr)
plt.show()


cv2.imshow('INVERTED IMAGE', new_img[10:, 28:145])
cv2.waitKey(0)
cv2.destroyAllWindows()



