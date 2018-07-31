#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created by zjj421 on 18-7-31
# Task: 
# Insights: 

from datetime import datetime
from functools import wraps

import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

from tqdm import tqdm


def erode_dilate_op():
    # img = cv2.imread("/home/topsky/Desktop/04+251mask.tif", cv2.IMREAD_GRAYSCALE)
    img = cv2.imread("/home/topsky/Desktop/04+246mask.tif", cv2.IMREAD_GRAYSCALE)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 定义结构元素
    # opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)  # 开运算
    result = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)  # 闭运算
    result = np.concatenate([img, result], axis=0)
    plt.imshow(result, "gray")
    plt.show()


def fill_contour(image_contour):
    """
    填充轮廓。
    :param image_contour: str or 2-d numpy array
    :return:
    """
    if isinstance(image_contour, str):
        img_in = cv2.imread(image_contour, cv2.IMREAD_GRAYSCALE)
    else:
        img_in = image_contour
    th, im_th = cv2.threshold(img_in, 127, 255, cv2.THRESH_BINARY)
    im_floodfill = im_th.copy()
    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_th.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255)
    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    # Combine the two images to get the foreground.
    im_out = im_th | im_floodfill_inv
    return im_out


def do_convert_custom_label2contour():
    test_image_dir = ""
    custom_label_image_dir = ""
    contour_save_dir = ""
    file_lst = next(os.walk(custom_label_image_dir))[2]
    contour_file_path_lst = [os.path.join(custom_label_image_dir, x) for x in file_lst]
    test_image_file_path_lst = [os.path.join(test_image_dir, x) for x in file_lst]
    image_contour_file_path_lst = [os.path.join(contour_save_dir, x) for x in file_lst]
    for i in range(len(contour_file_path_lst)):

        img_ori = cv2.imread(test_image_file_path_lst[i], cv2.IMREAD_UNCHANGED)
        img_contour = cv2.imread(contour_file_path_lst[i], cv2.IMREAD_UNCHANGED)[..., :3]

        is_same = img_ori == img_contour
        is_same = is_same * 1

        h, w, c = img_contour.shape
        img_contour = np.zeros(shape=(h, w))
        for i in range(h):
            for j in range(w):
                sum_ = sum(is_same[i, j])
                if sum_ != 3:
                    img_contour[i, j] = 255
        cv2.imwrite(image_contour_file_path_lst[i], img_contour)


def do_conver_contour2mask():
    contour_dir = ""
    mask_save_dir = ""
    file_lst = next(os.walk(contour_dir))[2]
    contour_file_path_lst = [os.path.join(contour_dir, x) for x in file_lst]
    mask_file_path_lst = [os.path.join(mask_save_dir, x) for x in file_lst]
    for i in range(len(contour_file_path_lst)):
        mask = fill_contour(contour_file_path_lst[i])
        cv2.imwrite(mask_file_path_lst[i], mask)
    print("Done.")


def rotate(image, angle):
    # if angle == 0 or -0:
    #     return image
    # height, width = image.shape[0:2]
    # mat = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    # img = cv2.warpAffine(image, mat, (width, height),
    #                      flags=cv2.INTER_LINEAR,
    #                      borderMode=cv2.BORDER_REFLECT_101)
    # img = np.expand_dims(img, -1)

    return image


def hflip(img, is_flip=1):
    if is_flip:
        return np.expand_dims(cv2.flip(img, 1), -1)
    else:
        return img


def tta_predict_on_batch(model, image):
    """

    :param image: 3-d numpy array, pre-processed image
    :return: 3-d numpy array, (h, w, c)
    """

    # rotate_angle_lst = [0, 4, 8, 12]
    rotate_angle_lst = [0]
    h_flip_lst = [0, 1]
    aug_lst = [(is_h_flip, rotate_angle) for is_h_flip in h_flip_lst
               for rotate_angle in rotate_angle_lst]
    preds_0 = []
    preds_1 = []
    for aug in tqdm(aug_lst):
        img = image.copy()
        img = hflip(img, aug[0])
        print(img.shape)
        img = rotate(img, aug[1])
        print(img.shape)
        img = np.expand_dims(img, 0)
        print(img.shape)
        pred = model.predict(img, batch_size=1)
        print(aug, "predicted")
        pred = np.squeeze(pred, axis=0)
        pred = rotate(pred, -aug[1])
        pred = hflip(pred, aug[0])
        pred_0 = pred[..., 0]
        pred_1 = pred[..., 1]
        preds_0.append(pred_0)
        preds_1.append(pred_1)
    preds_0 = np.mean(preds_0, axis=0)
    preds_1 = np.mean(preds_1, axis=1)
    preds = np.concatenate([preds_0, preds_1], axis=0)
    preds = np.transpose(preds, axes=[1, 2, 0])
    return preds


# not use
def vflip(img):
    return cv2.flip(img, 0)


# not use
def random_flip(img, code):
    return cv2.flip(img, code)


# not use
def clip(img, dtype, maxval):
    return np.clip(img, 0, maxval).astype(dtype)


# not use
def clipped(func):
    @wraps(func)
    def wrapped_function(img, *args, **kwargs):
        dtype, maxval = img.dtype, np.max(img)
        return clip(func(img, *args, **kwargs), dtype, maxval)

    return wrapped_function


# not use
@clipped
def random_brightness(img, alpha):
    return alpha * img


# not use
@clipped
def random_contrast(img, alpha):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = (3.0 * (1.0 - alpha) / gray.size) * np.sum(gray)
    return alpha * img + gray


def __main():
    np.set_printoptions(threshold=np.inf)
    # erode_dilate_op()
    # compare_image()
    pass


if __name__ == '__main__':
    start = datetime.now()
    print("Start time is {}".format(start))
    __main()
    end = datetime.now()
    print("End time is {}".format(end))
    print("\nTotal running time is {}s".format((end - start).seconds))
    print("\nCongratulations!!!")
