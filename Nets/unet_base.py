#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-11-21 上午10:53
# @Author  : Yu-dong Wu
# @Site    : 
# @File    : unet_base.py
# @Software: PyCharm
# @profile : original structure in unet
import tensorflow as tf

slim = tf.contrib.slim


def unet_base(inputs, output_stride=16):
    """
    encoder part of unet according to original paper
    :param inputs:
    :param output_stride: if output_stride < 16, dilated conv will be used
    :return:
    """
    end_points = {}
    stride2feat = {}
    current_stride = 1
    rate = 1

    assert output_stride <= 16, 'output_stride: %d should no more than 16' % output_stride
    with tf.variable_scope('unet', 'unet', [inputs]) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        net = inputs
        for i in range(4):
            if rate == 1:
                net = slim.repeat(net, 2, slim.conv2d, 64 * 2 ** i, [3, 3], scope='conv%d' % i)
                end_points['conv%d' % (i + 1)] = net
            else:
                net = slim.repeat(net, 2, slim.conv2d_transpose, 64 * 2 ** i, [3, 3], stride=rate, scope='dilated_conv%d' % i)
                end_points['dilated_conv%d' % (i + 1)] = net
            stride2feat['stride%d' % current_stride] = net

            if current_stride < output_stride:
                net = slim.max_pool2d(net, [2, 2], scope='pool%d' % i)
                end_points['pool%d' % (i + 1)] = net
                current_stride *= 2
            else:
                rate *= 2

        # 3x3
        net = slim.conv2d(net, 1024, [3, 3], stride=1, scope='conv5')
        end_points['conv5'] = net

    return net, end_points, stride2feat


def unet_base_arg_scope(is_training=True, weight_decay=0.00004, stddev=0.09):
    """
    defines the default unet arg scope
    :param is_training:
    :param weight_decay:
    :param stddev:
    :return:
    """
    batch_norm_params = {
        'is_training': is_training,
        'center': True,
        'scale': True,
        'decay': 0.9997,
        'epsilon': 0.001,
    }
    # Set weight_decay for weights in Conv and DepthSepConv layers.
    weights_init = tf.truncated_normal_initializer(stddev=stddev)
    regularizer = tf.contrib.layers.l2_regularizer(weight_decay)

    with slim.arg_scope([slim.conv2d], weights_initializer=weights_init,
                        activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm,
                        weights_regularizer=regularizer, padding='SAME'):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params) as sc:
            return sc