#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-4-9 上午9:55
# @Author  : Yu-dong Wu
# @Site    : 
# @File    : Metric.py
# @Software: PyCharm
# @profile :
import numpy as np
from scipy import spatial
import skimage
from skimage import io, morphology, img_as_bool
from collections import namedtuple
import os
MetricResults = namedtuple('MetricResults', ['dist_mean', 'dist_std', 'dist_median',
                                             'false', 'missing', 'F1'])


def distance_left_to_right(left_label, right_label):
    """

    :param left_label: label image of guide wire
    :param right_label: label image of guide wire
    :return: distances mean(min||left(pixel) - right(pixel)||2)
    """
    assert left_label.shape == right_label.shape, "left_label shape not equal right_label shape"

    # use a KDTree to calculate the nearest point
    y_array, x_array = np.where(left_label != 0)
    left_points = np.column_stack((x_array.copy(), y_array.copy()))

    y_array, x_array = np.where(right_label != 0)
    right_points = np.column_stack((x_array.copy(), y_array.copy()))

    KDTree = spatial.KDTree(right_points)
    distances, indexs = KDTree.query(left_points)
    return distances


def DistancesFromLeftToRightsMultis(left, right, skeletonize=True):
    """
    calculate min distances from left to right use KDtree
    :param left: [n, h, w, 1]
    :param right: [n, h, w, 1]
    :return: [[]*n]
    """
    num = left.shape[0]
    distances_l_r = []
    distances_r_l = []
    for i in range(num):
        if skeletonize:
            l = Skeletonize(np.squeeze(left[i]))
            r = Skeletonize(np.squeeze(right[i]))
        else:
            l = np.squeeze(left[i])
            r = np.squeeze(right[i])
        distances_l_r.append(distance_left_to_right(l, r))
        distances_r_l.append(distance_left_to_right(r, l))

    return distances_l_r, distances_r_l


def Skeletonize(grey_img):
    """
    skeletonize grey image has shape [h, w] not [h, w, 1]
    :param grey_img: dtype need to be uint8
    :return: image in uint8
    """
    img_bool = img_as_bool(grey_img)
    img_sk = morphology.medial_axis(img_bool)
    img_sk = skimage.img_as_ubyte(img_sk)
    # print(np.max(img_sk))
    return img_sk


def DistancesToMetrics(distances_l_r, distances_r_l):
    thr_false = 3.0
    thr_miss = 3.0
    dist_means = np.array([i.mean() for i in distances_l_r])

    false_percent = []
    for dists in distances_l_r:
        false_percent.append(sum(dists > thr_false)/dists.shape[0])
    false_percent = np.array(false_percent)

    miss_percent = []
    for dists in distances_r_l:
        miss_percent.append(sum(dists > thr_miss) / dists.shape[0])
    miss_percent = np.array(miss_percent)

    recall = 1.0 - miss_percent
    precision = 1.0 - false_percent
    F1 = 2.0 * (recall * precision) / (recall + precision + 1e-7)

    return MetricResults(dist_means.mean(), dist_means.std(), np.median(dist_means),
                         false_percent.mean(), miss_percent.mean(), F1.mean())