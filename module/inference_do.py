#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created by zjj421 on 18-7-28
# Task: 
# Insights: 

import os
from datetime import datetime
import keras.backend as K
import cv2
import numpy as np
from func.data_io import DataSet
from func.model_densenet import get_densenet121_unet_sigmoid_gn
from func.model_inception_resnet_v2_gn import get_inception_resnet_v2_unet_sigmoid_gn
from func.model_se_inception_resnet_v2_gn import get_se_inception_resnet_v2_unet_sigmoid_gn
from tqdm import tqdm

from module.competition_utils import ensemble_from_pred, convert_submission, IMAGE_FILE_LIST, MASK_FILE_LIST, \
    TEST_DATA_DIR
from module.inference_base import ModelDeployment, get_acc
from module.utils_public import get_file_path_list


def do_get_acc():
    h5_data_path = "/home/zj/helloworld/study/njai_challenge/cbct/inputs/data_0717.hdf5"
    val_fold_nb = 2
    output_channels = 2
    is_train = False
    model_def = get_inception_resnet_v2_unet_sigmoid_gn(weights=None, output_channels=output_channels)
    model_weights = "/home/zj/helloworld/study/njai_challenge/cbct/model_weights/final_inception_resnet_v2_gn_fold2_1i_2o.h5"
    get_acc(model_def, model_weights, h5_data_path, val_fold_nb, is_train)


def infer_do():
    val_fold_nb = "01"
    h5_data_path = "/home/topsky/helloworld/study/njai_challenge/cbct/inputs/data_0717.hdf5"
    result_save_dir = "/home/topsky/helloworld/study/njai_challenge/cbct/result/image_mask_preds20180730"
    model_weights = "/home/topsky/helloworld/study/njai_challenge/cbct/model_weights/20180730_1/best_val_loss_densenet_bn_fold01_1i_2o_20180730.h5"
    model_def = get_densenet121_unet_sigmoid(weights=None, output_channels=2)
    model_obj = ModelDeployment(model_def, model_weights)
    model_obj.predict_from_h5data(h5_data_path, val_fold_nb=val_fold_nb, is_train=False, save_dir=result_save_dir)
    model_obj.predict_from_h5data(h5_data_path, val_fold_nb=val_fold_nb, is_train=True, save_dir=result_save_dir)

    # dataset = DataSet(h5_data_path, val_fold_nb=1)
    # keys = dataset.val_keys
    # images = dataset.get_image_by_key(keys[8])
    # images = np.expand_dims(images, axis=0)
    # images = np.expand_dims(images, axis=-1)
    # model_obj.predict_and_show(images, show_output_channels=2)


def do_evaluate():
    h5_data_path = "/home/topsky/helloworld/study/njai_challenge/cbct/inputs/data_0717.hdf5"
    val_fold_nb = "01"
    output_channels = 2
    is_train = False
    model_def = get_se_inception_resnet_v2_unet_sigmoid_gn(weights=None, output_channels=output_channels)
    model_weights = "/home/topsky/helloworld/study/njai_challenge/cbct/model_weights/20180728_1/best_val_loss_se_inception_resnet_v2_gn_fold01_1i_2o_20180728.h5"

    model_obj = ModelDeployment(model_def, model_weights)
    dataset = DataSet(h5_data_path, val_fold_nb=val_fold_nb)
    images, masks = dataset.prepare_1i_2o_data(is_train=is_train)
    print(images.shape)
    print(masks.shape)
    # idx_lst = [0, 5, 10, 15, 20]
    # val_images = np.array([images[i] for i in idx_lst])
    # val_masks = np.array([masks[i] for i in idx_lst])
    # model_obj.evaluate(val_images, val_masks)
    model_obj.evaluate(images, masks)
    # get_acc(model_def, model_weights, h5_data_path, val_fold_nb, is_train)


def make_sub():
    mask_file_list = ['03+261mask.tif', '03+262mask.tif', '04+246mask.tif', '04+248mask.tif', '04+251mask.tif']
    test_result_dir = "/media/topsky/HHH/jzhang_root/data/njai/cbct/CBCT_testingset/CBCT_testingset_pred20180731_0/result_1/mask0"
    sub_json_file_path = "/home/topsky/helloworld/study/njai_challenge/submissions/CBCT_testingset_pred20180731_0_result_1.json"
    dst_mask_file_path_lst = [os.path.join(test_result_dir, x) for x in mask_file_list]
    convert_submission(dst_mask_file_path_lst, sub_json_file_path)


# TODO 假定输入图片大小一样，不一样怎么办？ 后续把函数拆分实现。
def inference_and_sub():
    model_def_lst1 = [
                         get_se_inception_resnet_v2_unet_sigmoid_gn
                     ] * 21
    model_def_lst2 = [
                         get_densenet121_unet_sigmoid_gn
                     ] * 21
    model_def_lst1.extend(model_def_lst2)
    model_def_lst = model_def_lst1
    model_weights_dir1 = "/home/topsky/helloworld/study/njai_challenge/cbct/model_weights/20180801_select/inception_resnet_v2"
    model_weights_lst1 = get_file_path_list(model_weights_dir1, ext=".h5")
    model_weights_dir2 = "/home/topsky/helloworld/study/njai_challenge/cbct/model_weights/20180801_select/densenet"
    model_weights_lst2 = get_file_path_list(model_weights_dir2, ext=".h5")
    model_weights_lst1.extend(model_weights_lst2)
    model_weights_lst = model_weights_lst1

    assert len(model_def_lst) == len(model_weights_lst)

    sub_json_file_path = "/home/topsky/helloworld/study/njai_challenge/submissions/best_val_loss_sub_20180801_0_ensemble42_c2.json"

    test_result_dir = "/media/topsky/HHH/jzhang_root/data/njai/cbct/CBCT_testingset/CBCT_testingset_pred20180801_42"

    image_file_path_lst = [os.path.join(TEST_DATA_DIR, x) for x in IMAGE_FILE_LIST]
    dst_mask_file_path_lst = [os.path.join(test_result_dir, x) for x in MASK_FILE_LIST]
    preds = []
    for i, (model_def, model_weights) in enumerate(zip(model_def_lst, model_weights_lst)):
        print("predicting ...", i + 1)
        if i == 0:
            model_obj = ModelDeployment(model_def(input_shape=(None, None, 1), weights=None, output_channels=2),
                                        model_weights)
        else:
            try:
                model_obj.reload_weights(model_weights)
            except:
                del model_obj
                model_obj = ModelDeployment(model_def(input_shape=(None, None, 1), weights=None, output_channels=2),
                                            model_weights)
        sub_result_save_dir_basename = os.path.splitext(os.path.basename(model_weights))[0]
        sub_result_save_dir = os.path.join(test_result_dir, sub_result_save_dir_basename)

        pred = model_obj.predict_from_files(image_file_path_lst,
                                            batch_size=5,
                                            mask_file_lst=MASK_FILE_LIST,
                                            tta=True,
                                            is_save_npy=True,
                                            is_save_mask0=True,
                                            is_save_mask1=True,
                                            result_save_dir=sub_result_save_dir
                                            )  # (b, h, w, c)
        preds.append(pred)

    preds = np.array(preds)  # (nb_model, b, h, w, c)

    preds_0 = preds[..., 0]  # (nb_model, b, h, w)
    preds_1 = preds[..., 1]  # (nb_model, b, h, w)

    pred_0 = []
    pred_1 = []
    for b in range(len(image_file_path_lst)):
        pred_b_0 = preds_0[:, b, :, :]
        pred_b_0 = np.mean(pred_b_0, axis=0)
        pred_0.append(pred_b_0)

        pred_b_1 = preds_1[:, b, :, :]
        pred_b_1 = np.mean(pred_b_1, axis=0)
        pred_1.append(pred_b_1)
    pred_0 = np.array(pred_0)
    pred_1 = np.array(pred_1)
    final_npy_save_dir = os.path.join(test_result_dir, "final_npy")
    final_preds = np.concatenate([np.expand_dims(pred_0, -1), np.expand_dims(pred_1, -1)], axis=-1)
    ModelDeployment.save_npy(final_preds, MASK_FILE_LIST, final_npy_save_dir)

    pred_0 = np.where(pred_0 > 0.5, 255, 0)
    pred_1 = np.where(pred_1 > 0.5, 1, 0)
    pred_0 = pred_0 * pred_1
    for i in range(len(pred_0)):
        cv2.imwrite(dst_mask_file_path_lst[i], pred_0[i])
    convert_submission(dst_mask_file_path_lst, sub_json_file_path)


def __main():
    K.set_learning_phase(0)
    # make_sub()
    # tta_inference_and_sub()
    inference_and_sub()
    # do_evaluate()
    # infer_do()
    pass


if __name__ == '__main__':
    start = datetime.now()
    print("Start time is {}".format(start))
    __main()
    end = datetime.now()
    print("End time is {}".format(end))
    print("\nTotal running time is {}s".format((end - start).seconds))
    print("\nCongratulations!!!")
