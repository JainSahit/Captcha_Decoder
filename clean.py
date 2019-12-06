import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
kernel = np.ones((3,2),np.uint8)
#np.set_printoptions(threshold=sys.maxsize)

data_path = "/Users/sahitjain/Desktop/Fall-19/IFT-598/Project/captcha_dataset/samples"

for img in os.listdir(data_path):
    iarr = cv2.imread(os.path.join(data_path, img), cv2.IMREAD_GRAYSCALE)

print(len(iarr))

training_data = []
img = cv2.imread("/Users/sahitjain/Desktop/Fall-19/IFT-598/Project/captcha_dataset/samples/wbncw.png", 0)
#img = img[10:, 28:145]
print(img)
nIMG = np.array(img)
print(nIMG.shape)
inverted_img = (255.0 - img)
new_img = inverted_img / 255.0
print(new_img)

ret,thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
ret,thresh6 = cv2.threshold(img, 127, 255, cv2.THRESH_OTSU)
dilate1 = cv2.dilate(thresh6, kernel, iterations=1)
errosion1 = cv2.erode(dilate1, kernel, iterations=1)
opening = cv2.morphologyEx(errosion1, cv2.MORPH_OPEN, kernel)
titles = ['Original Image','BINARY','OTSU','MORPH OPENING', 'DILATE', 'ERODE']
images = [img, thresh1, thresh6, opening, dilate1, errosion1]

for i in range(9):
    plt.subplot(3,3,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()


#get the bounding rect 1233

img1 = opening[10:, 28:61]
img2 = opening[10:, 62:78]
img3 = opening[10:, 70:93]
img4 = opening[10:, 93:115]
img5 = opening[10:, 116:145]

titles3 = ['Character1', 'Character2', 'Character3', 'Character4', 'Character5', 'Original']
images3 = [img1, img2, img3, img4, img5, opening]

for i in range(6):
    plt.subplot(3, 2, i + 1), plt.imshow(images3[i], 'gray')
    plt.title(titles3[i])
    plt.xticks([]), plt.yticks([])

plt.show()
#plt.hist(opening,256,[0,256]),plt.show()
plt.hist(opening.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')
histr = cv2.calcHist([img],[0],None,[256],[0,256])
plt.plot(histr)
plt.show()


#cv2.imshow('INVERTED IMAGE', new_img[10:, 28:145])
cv2.waitKey(0)
cv2.destroyAllWindows()

"""

n_digits = 19

kmeans = MiniBatchKMeans(n_clusters=n_digits)

kmeans.fit(reX)

centroids = kmeans.cluster_centers_
print(centroids.shape)
images = centroids.reshape(19, 40, 117)
images *= 255
images = images.astype(np.uint8)

fig, axs = plt.subplots(6, 6, figsize=(20, 20))
plt.gray()

for i, ax in enumerate(axs.flat):
    # determine inferred label using cluster_labels dictionary
    for key, value in cluster_labels.items():
        if i in value:
            ax.set_title('Inferred Label: {}'.format(key))

    # add image to subplot
    ax.matshow(images[i])
    ax.axis('off')
fig.show()

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
k = 19
attempts = 10

ret, label, center = cv2.kmeans(reY[:20], k, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
centre = np.uint8(center)
res = centre[label.flatten()]
result_image = res.reshape(nY[1].shape)

figure_size = 15
plt.figure(figsize=(figure_size, figure_size))
plt.subplot(1, 2, 1), plt.imshow(nX[1], 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(1, 2, 2), plt.imshow(result_image, 'gray')
plt.title('Segmented Image when K = %i' % k), plt.xticks([]), plt.yticks([])
plt.show()


titles = ['Original Image', 'Character 1', 'Character 2', 'Character 3', 'Character 4', 'Character 5']
images = [nX[1], Y[5], Y[6], Y[7], Y[8], Y[9]]

for i in range(6):
    plt.subplot(3, 2, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()

"""



