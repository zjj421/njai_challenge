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
from func.model_se_inception_resnet_v2_gn import get_se_inception_resnet_v2_unet_sigmoid_gn

from module.competition_utils import ensemble_from_pred, convert_submission, IMAGE_FILE_LIST, MASK_FILE_LIST, \
    TEST_DATA_DIR
from module.inference_base import ModelDeployment, get_acc


def do_get_acc():
    h5_data_path = "/home/zj/helloworld/study/njai_challenge/cbct/inputs/data_0717.hdf5"
    val_fold_nb = 2
    output_channels = 2
    is_train = False
    model_def = get_inception_resnet_v2_unet_sigmoid_gn(weights=None, output_channels=output_channels)
    model_weights = "/home/zj/helloworld/study/njai_challenge/cbct/model_weights/final_inception_resnet_v2_gn_fold2_1i_2o.h5"
    get_acc(model_def, model_weights, h5_data_path, val_fold_nb, is_train)


def infer_do():
    h5_data_path = "/home/zj/helloworld/study/njai_challenge/cbct/inputs/data_0717.hdf5"
    result_save_dir = "/media/zj/share/data/njai_2018/cbct/result/image_mask_preds20180727"
    model_weights = "/home/zj/helloworld/study/njai_challenge/cbct/model_weights/final_inception_resnet_v2_gn_fold2_1i_2o.h5"
    model_def = get_inception_resnet_v2_unet_sigmoid_gn(weights=None, output_channels=2)
    model_obj = ModelDeployment(model_def, model_weights)
    model_obj.predict_from_h5data(h5_data_path, val_fold_nb=2, is_train=False, save_dir=result_save_dir)
    model_obj.predict_from_h5data(h5_data_path, val_fold_nb=2, is_train=True, save_dir=result_save_dir)

    # dataset = DataSet(h5_data_path, val_fold_nb=1)
    # keys = dataset.val_keys
    # images = dataset.get_image_by_key(keys[8])
    # images = np.expand_dims(images, axis=0)
    # images = np.expand_dims(images, axis=-1)
    # model_obj.predict_and_show(images, show_output_channels=2)


def do_evaluate():
    h5_data_path = "/home/topsky/helloworld/study/njai_challenge/cbct/inputs/data_0717.hdf5"
    val_fold_nb = 1
    output_channels = 2
    is_train = False
    model_def = get_se_inception_resnet_v2_unet_sigmoid_gn(weights=None, output_channels=output_channels)
    model_weights = "/home/topsky/helloworld/study/njai_challenge/cbct/model_weights/best_val_acc_se_inception_resnet_v2_gn_fold1_1i_2o_20180726.h5"

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


def do_inference_2stages():
    model_def_stage1 = get_inception_resnet_v2_unet_sigmoid_gn_old(input_shape=(576, 576, 1), weights=None,
                                                                   output_channels=1)
    model_weights_stage1 = "/home/topsky/helloworld/study/njai_challenge/cbct/model_weights/inception_resnet_v2_gn_fold1_1i_1o_stage1.h5"
    model_def_stage2 = get_inception_resnet_v2_unet_sigmoid_gn_old(input_shape=(576, 576, 2), weights=None,
                                                                   output_channels=1)
    model_weights_stage2 = "/home/topsky/helloworld/study/njai_challenge/cbct/model_weights/inception_resnet_v2_gn_fold1_2i_1o_stage2_20180726.h5"
    file_dir = "/media/topsky/HHH/jzhang_root/data/njai/cbct/CBCT_testingset/CBCT_testingset"
    result_save_dir = "/media/topsky/HHH/jzhang_root/data/njai/cbct/CBCT_testingset/CBCT_testingset_2stages_pred20180727"
    inference_2stages_from_files(model_def_stage1, model_weights_stage1, model_def_stage2, model_weights_stage2,
                                 file_dir, result_save_dir)


def make_sub():
    mask_file_list = ['03+261mask.tif', '03+262mask.tif', '04+246mask.tif', '04+248mask.tif', '04+251mask.tif']
    test_result_dir = "/media/topsky/HHH/jzhang_root/data/njai/cbct/CBCT_testingset/CBCT_testingset_pred20180728_1/result_2"
    sub_json_file_path = "/home/topsky/helloworld/study/njai_challenge/submissions/CBCT_testingset_pred20180728_1_result_2.json"
    dst_mask_file_path_lst = [os.path.join(test_result_dir, x) for x in mask_file_list]
    convert_submission(dst_mask_file_path_lst, sub_json_file_path)


def inference_and_sub():
    model_def_lst = [
        get_se_inception_resnet_v2_unet_sigmoid_gn,
        get_se_inception_resnet_v2_unet_sigmoid_gn,
        get_se_inception_resnet_v2_unet_sigmoid_gn,
        get_se_inception_resnet_v2_unet_sigmoid_gn,
        get_se_inception_resnet_v2_unet_sigmoid_gn,
        get_se_inception_resnet_v2_unet_sigmoid_gn,
        get_se_inception_resnet_v2_unet_sigmoid_gn,
        get_se_inception_resnet_v2_unet_sigmoid_gn,
    ]
    model_weights_lst = [
        "/home/topsky/helloworld/study/njai_challenge/cbct/model_weights/20180728_1/best_val_loss_se_inception_resnet_v2_gn_fold01_1i_2o_20180728.h5",
        "/home/topsky/helloworld/study/njai_challenge/cbct/model_weights/20180728_1/best_val_loss_se_inception_resnet_v2_gn_fold02_1i_2o_20180728.h5",
        "/home/topsky/helloworld/study/njai_challenge/cbct/model_weights/20180728_1/best_val_loss_se_inception_resnet_v2_gn_fold11_1i_2o_20180728.h5",
        "/home/topsky/helloworld/study/njai_challenge/cbct/model_weights/20180728_1/best_val_loss_se_inception_resnet_v2_gn_fold12_1i_2o_20180728.h5",
        "/home/topsky/helloworld/study/njai_challenge/cbct/model_weights/20180728_1/best_val_loss_se_inception_resnet_v2_gn_fold13_1i_2o_20180728.h5",
        "/home/topsky/helloworld/study/njai_challenge/cbct/model_weights/20180728_1/best_val_loss_se_inception_resnet_v2_gn_fold21_1i_2o_20180728.h5",
        "/home/topsky/helloworld/study/njai_challenge/cbct/model_weights/20180728_1/best_val_loss_se_inception_resnet_v2_gn_fold22_1i_2o_20180728.h5",
        "/home/topsky/helloworld/study/njai_challenge/cbct/model_weights/20180728_1/best_val_loss_se_inception_resnet_v2_gn_fold23_1i_2o_20180728.h5",
    ]
    sub_json_file_path = "/home/topsky/helloworld/study/njai_challenge/submissions/best_val_loss_sub_20180729_0_ensemble8.json"
    test_result_dir = "/media/topsky/HHH/jzhang_root/data/njai/cbct/CBCT_testingset/CBCT_testingset_pred20180729_0"

    if not os.path.isdir(test_result_dir):
        os.makedirs(test_result_dir)

    image_file_path_lst = [os.path.join(TEST_DATA_DIR, x) for x in IMAGE_FILE_LIST]
    dst_mask_file_path_lst = [os.path.join(test_result_dir, x) for x in MASK_FILE_LIST]
    masks = []
    for i, (model_def, model_weights) in enumerate(zip(model_def_lst, model_weights_lst)):
        if i == 0:
            model_obj = ModelDeployment(model_def(input_shape=(None, None, 1), weights=None, output_channels=2),
                                        model_weights)
        else:
            model_obj.reload_weights(model_weights)

        sub_result_save_dir = os.path.join(test_result_dir, "result_{}".format(i))
        if not os.path.isdir(sub_result_save_dir):
            os.makedirs(sub_result_save_dir)
        # TODO 没有利用第二个输出通道
        pred = model_obj.predict_from_files(image_file_path_lst,
                                            result_save_dir=sub_result_save_dir,
                                            mask_file_lst=MASK_FILE_LIST
                                            )  # (5, 576, 576)
        print(pred.shape)
        masks.append(pred)
    masks = np.transpose(masks, axes=[1, 2, 3, 0])
    for i in range(len(masks)):
        pred_ensemble = ensemble_from_pred(masks[i], threshold=0.5)
        cv2.imwrite(dst_mask_file_path_lst[i], pred_ensemble)
    convert_submission(dst_mask_file_path_lst, sub_json_file_path)


def __main():
    K.set_learning_phase(0)
    # make_sub()
    inference_and_sub()

    pass


if __name__ == '__main__':
    start = datetime.now()
    print("Start time is {}".format(start))
    __main()
    end = datetime.now()
    print("End time is {}".format(end))
    print("\nTotal running time is {}s".format((end - start).seconds))
    print("\nCongratulations!!!")
