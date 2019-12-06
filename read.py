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
            img = img[10:, 28:145]
            ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
            ret, thresh2 = cv2.threshold(thresh1, 127, 255, cv2.THRESH_OTSU)
            dilate = cv2.dilate(thresh2, kernel, iterations=1)
            erosion = cv2.erode(dilate, kernel, iterations=1)
            opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel)
            opening = opening / 255.0
            images.append(opening)

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
def verticalProjection(img):
    #"Return a list containing the sum of the pixels in each column"
    (h, w) = img.shape[:2]
    sumCols = []
    for j in range(w):
        col = img[0:h, j:j + 1]  # y1:y2, x1:x2
        sumCols.append(np.sum(col))
    return sumCols


def vP(img):
    image = []
    for i in range(0, len(img)):
        image.append(verticalProjection(nY[i]))

    return np.array(image)


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


"""
tsne_results = np.load(wd + "trainD1.npy")
model = dill.load(open(wd + "model1", "rb"))
Y = model.predict(tsne_results)
#print(model.cluster_centers_)
print(Y[726])
print(model.labels_[726])


tst, lbl = load_images_from_folder(test)
ntst = np.float32(tst)
ytst, tst_label = segment_image(tst, lbl)
nytst = np.float32(ytst)
reYtst = nytst.reshape(len(nytst), -1)
reYtst = np.int64(reYtst)
tsne_tst = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results_tst = tsne_tst.fit_transform(reYtst)

Ytst = model.predict(tsne_results_tst)
print(Ytst)
"""
