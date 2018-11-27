#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-11-20 下午7:21
# @Author  : Yu-dong Wu
# @Site    : 
# @File    : network_common.py
# @Software: PyCharm
# @profile : some tools
import tensorflow as tf
import math
slim = tf.contrib.slim


def get_backbone_func_and_arg_scope(config):
    """

    :param network_name: string
    :return: network func
    """
    if config.Network.encoder_backbone is None:
        return None
    elif config.Network.encoder_backbone == 'MobilenetV1':
        from mobilenet_v1 import mobilenet_v1_base, mobilenet_v1_arg_scope
        return mobilenet_v1_base, mobilenet_v1_arg_scope(is_training=config.is_training)
    elif config.Network.encoder_backbone == 'Ori_Unet':
        from unet_base import unet_base, unet_base_arg_scope
        return unet_base, unet_base_arg_scope(is_training=config.is_training)
    else:
        raise Exception('invalid network name: %s' % config.Network.encoder_backbone)


def decoder_func(strides2feat, config):
    """
    generate decoder backbone automatically according to the encoder backbone
    Conv, transeConv, Concat, Conv
    :param strides2feat:
    :param config:
    :return:
    """
    max_stride = config.Network.max_stride
    feat = strides2feat['stride%d' % max_stride]

    with slim.arg_scope([slim.conv2d_transpose, slim.conv2d], activation_fn=tf.nn.relu,
                        weights_regularizer=slim.l2_regularizer(config.Train.weight_decay),
                        biases_initializer=tf.zeros_initializer(), normalizer_fn=slim.batch_norm,
                        padding='SAME', ):
        # fisrt conv: keep depth, kernel 3*3
        depth = feat.get_shape().as_list()[-1]
        feat = slim.conv2d(feat, depth, [3, 3], stride=1)

        # decoder:
        for i in range(int(math.log(max_stride, 2)), 0, -1):
            # conv: keep channel
            depth = feat.get_shape().as_list()[-1]
            feat = slim.conv2d(feat, depth, [3, 3], stride=1)
            # transeConv: stride 2, channel decrease to half
            depth = strides2feat['stride%d' % 2**(i-1)].get_shape().as_list()[-1]
            feat = slim.conv2d_transpose(feat, depth, [3, 3], stride=2)
            # concat
            feat = tf.concat([feat, strides2feat['stride%d' % 2**(i-1)]], axis=3)
            # conv : channel decrease to half
            feat = slim.conv2d(feat, depth, [3, 3], stride=1)

        depth = feat.get_shape().as_list()[-1]
        feat = slim.conv2d(feat, depth, [3, 3], stride=1)

        return feat
