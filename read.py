import cv2
import numpy as np
import os
import sys
import time
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import dill

np.set_printoptions(threshold=sys.maxsize)
kernel = np.ones((3, 2), np.uint8)
wd = "/Users/sahitjain/Desktop/Fall-19/IFT-598/Project/Data/"
data_path = "/Users/sahitjain/Desktop/Fall-19/IFT-598/Project/captcha_dataset/samples"



def load_images_from_folder(folder):
    images = []
    pic_target = []
    for imagename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, imagename), 0)
        if img is not None:
            pic_target.append(imagename[:-4])
            img1 = img
            img = img[10:, 28:145]
            ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
            ret, thresh2 = cv2.threshold(thresh1, 127, 255, cv2.THRESH_OTSU)
            dilate = cv2.dilate(thresh2, kernel, iterations=1)
            erosion = cv2.erode(dilate, kernel, iterations=1)
            opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel)
            opening = opening / 255.0
            images.append(opening)
            """
            titles = ['ORIGINAL IMAGE', 'SIZE REDUCTION', 'BINARY INVERSION', 'OTSU', 'DILATION', 'ERROSION', 'MORPHOLOGICAL OPENING']
            images = [img1, img, thresh1, thresh2, dilate, erosion, opening]

            for i in range(7):
                plt.subplot(4, 2, i + 1), plt.imshow(images[i], 'gray')
                plt.title(titles[i])
                plt.xticks([]), plt.yticks([])
            plt.show()
"""
    return images, pic_target


def write_image(img, label, c):
    ch = label + "_" + str(c) + ".png"
    image = img * 255.0
    cv2.imwrite(wd + "/SegmentedCharacters/" + ch, image)


def segment_image(img_arr, label):
    character_img = []
    character_lbl = []
    count = 0
    for i in range(0, len(img_arr)):
        ch1 = img_arr[i][:, 5:25]
        ch2 = img_arr[i][:, 24:44]
        ch3 = img_arr[i][:, 43:63]
        ch4 = img_arr[i][:, 63:83]
        ch5 = img_arr[i][:, 85:105]
        lbl1 = label[i][0]
        lbl2 = label[i][1]
        lbl3 = label[i][2]
        lbl4 = label[i][3]
        lbl5 = label[i][4]
        character_lbl.append(lbl1)
        character_lbl.append(lbl2)
        character_lbl.append(lbl3)
        character_lbl.append(lbl4)
        character_lbl.append(lbl5)
        character_img.append(ch1)
        character_img.append(ch2)
        character_img.append(ch3)
        character_img.append(ch4)
        character_img.append(ch5)
        write_image(ch1, lbl1, count)
        count += 1
        write_image(ch2, lbl2, count)
        count += 1
        write_image(ch3, lbl3, count)
        count += 1
        write_image(ch4, lbl4, count)
        count += 1
        write_image(ch5, lbl5, count)
        count += 1

    return character_img, character_lbl
"""

X, label = load_images_from_folder(data_path)


nX = np.float32(X)
Y, ch_label = segment_image(X, label)
nY = np.float32(Y)
#write_image(nY)
print(nY.shape)
print(len(Y))

reX = nX.reshape(len(nX), -1)

reY = nY.reshape(len(nY), -1)
reY = np.int64(reY)


y_sum = vP(nY)

y_sumReshape = y_sum[:, 1]

time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(reY)

np.save(wd + "trainD1", tsne_results)

np.save(wd + "labels1", ch_label)
"""



