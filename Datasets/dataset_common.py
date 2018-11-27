#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-11-26 下午7:34
# @Author  : Yu-dong Wu
# @Site    : 
# @File    : dataset_common.py
# @Software: PyCharm
# @profile : common utils for dataset
import tensorflow as tf

def count_samples_in(tfrecords):
    """
    count number of samples in tfrecords
    :param tfrecords: [tfrec1, tfrec2]
    :return:
    """
    count = 0
    if isinstance(tfrecords, str):
        tfrecords = [tfrecords]
    for tfrec in tfrecords:
        for rec in tf.python_io.tf_record_iterator(tfrec):
            count += 1
    return count