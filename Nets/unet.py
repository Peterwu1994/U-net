#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-4-3 上午11:23
# @Author  : Yu-dong Wu
# @Site    : 
# @File    : unet.py.py
# @Software: PyCharm
# @profile : implement unet using several different backbone
import tensorflow as tf

from Config.Config import config
from network_common import get_backbone_func_and_arg_scope, decoder_func

slim = tf.contrib.slim

def unet(inputs, config):
    """

    :param inputs: input data of network
    :param config:
    :return:
    """
    encoder_backbone_func, argsc = get_backbone_func_and_arg_scope(config)
    end_points = {}

    with slim.arg_scope(argsc):
        input_shape = inputs.get_shape().as_list()
        if len(input_shape) != 4:
            raise ValueError('Invalid input tensor rank, expected 4, was: %d' % len(input_shape))

        _, _, strides2feat = encoder_backbone_func(inputs, output_stride=config.Network.max_stride)
        feat = decoder_func(strides2feat, config)

        logits = slim.conv2d(feat, config.Dataset.seg_num_class, [3, 3], stride=1, normalizer_fn=None, scope='logits')
        end_points['logits'] = logits
        return logits, end_points


if __name__ == '__main__':
    data = tf.ones(shape=(1, 512, 512, 3), dtype=tf.float32)
    unet(data, config)





