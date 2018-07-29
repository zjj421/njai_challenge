#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created by zjj421 on 18-7-29
# Task: 
# Insights: 

from datetime import datetime
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
np.set_printoptions(threshold=np.inf)
print(check_output(["ls",
                    "/home/topsky/helloworld/Mask_RCNN/samples/nucleus/input/stage1_train_color/00ae65c1c6631ae6f2be1a449902976e6eb8483bf6b0740d00530220832c6d3e"]).decode(
    "utf8"))

import skimage.io
import skimage.segmentation
import matplotlib.pyplot as plt
from glob import glob

plt.rcParams['figure.figsize'] = 10, 10
# Any results you write to the current directory are saved as output.

# Load a single image and its associated masks
file = "/home/topsky/helloworld/Mask_RCNN/samples/nucleus/input/stage1_train_color/00ae65c1c6631ae6f2be1a449902976e6eb8483bf6b0740d00530220832c6d3e/images/00ae65c1c6631ae6f2be1a449902976e6eb8483bf6b0740d00530220832c6d3e.png"
masks = "/home/topsky/helloworld/Mask_RCNN/samples/nucleus/input/stage1_train_color/00ae65c1c6631ae6f2be1a449902976e6eb8483bf6b0740d00530220832c6d3e/masks/*.png"
image = skimage.io.imread(file)
masks = skimage.io.imread_collection(masks).concatenate()
print(type(masks))
print(masks.shape)
height, width, _ = image.shape
num_masks = masks.shape[0]

# Make a ground truth label image (pixel value is index of object label)
labels = np.zeros((height, width), np.uint16)
for index in range(0, num_masks):
    labels[masks[index] > 0] = index + 50


# plt.imshow(image)
# plt.imshow(labels, alpha=0.6)
# plt.show()
def getContour(mask):
    ### generate bound
    # plt.imshow(mask)
    print(mask.shape)
    mask_pad = np.pad(mask, ((1, 1), (1, 1)), 'constant')

    print(mask_pad.shape)

    h, w = mask.shape
    contour = np.zeros((h, w), dtype=np.bool)
    for i in range(3):
        for j in range(3):
            if i == j == 1:
                continue
            edge = (np.float32(mask) - np.float32(mask_pad[i:i + h, j:j + w])) > 0
            contour = np.logical_or(contour, edge)
    return contour


def showContour(image, contours):
    vis = np.copy(image)
    for contour in contours:
        vis[:, :, 0] ^= np.uint8(contour * 255)
    plt.imshow(vis)


contours = [getContour(mask) for mask in masks]
showContour(image, contours)

# plt.show()
# exit()

from scipy.ndimage.morphology import distance_transform_edt

edts = [distance_transform_edt(mask) for mask in masks]
plt.imshow(np.sum(edts, axis=0))

## contour finding with distance transform
plt.imshow(np.sum(edts, axis=0) == 1)

plt.show()

def __main():
    pass


if __name__ == '__main__':
    start = datetime.now()
    print("Start time is {}".format(start))
    __main()
    end = datetime.now()
    print("End time is {}".format(end))
    print("\nTotal running time is {}s".format((end - start).seconds))
    print("\nCongratulations!!!")
