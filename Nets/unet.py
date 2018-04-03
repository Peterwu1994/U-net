#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-4-3 上午11:23
# @Author  : Yu-dong Wu
# @Site    : 
# @File    : unet.py.py
# @Software: PyCharm
# @profile :
import tensorflow as tf
from Config import Config

slim = tf.contrib.slim



def Unet(inputs, config):
    """"""
    end_points = {}
    with tf.variable_scope('unet', 'unet', [inputs]) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        # encode
        with slim.arg_scope([slim.conv2d],
                            activation_fn=tf.nn.relu,
                            weights_regularizer=slim.l2_regularizer(config.weight_decay),
                            biases_initializer=tf.zeros_initializer(),
                            padding='SAME',
                            outputs_collections=end_points_collection):
            net = inputs
            for i in range(4):
                net = slim.repeat(net, 2, slim.conv2d, 64*2**i, [3, 3], scope='conv%d' % i)
                end_points['conv%d' % (i+1)] = net
                net = slim.max_pool2d(net, [2, 2], scope='pool%d' % i)
                end_points['pool%d' % (i+1)] = net

            # 3x3 *2
            net = slim.repeat(net, 2, slim.conv2d, 1024, [3, 3], scope='conv5')
            end_points['conv5'] = net

        # decode
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            activation_fn=tf.nn.relu,
                            weights_regularizer=slim.l2_regularizer(config.weight_decay),
                            biases_initializer=tf.zeros_initializer(),
                            padding='SAME',
                            outputs_collections=end_points_collection):

            for i in range(1, 5):
                channel_num = int(1024/2**i)
                net = slim.conv2d_transpose(net, channel_num, [3, 3], stride=2, scope='deconv%d'%i)
                end_points['deconv%d'%i] = net
                net = tf.concat([net, end_points['conv%d'%(5-i)]], axis=3)
                net = slim.repeat(net, 2, slim.conv2d, channel_num, [3, 3], scope='conv%d'%(i+5))
                end_points['conv%d' % (i+5)] = net

            logits = slim.conv2d(net, 2, [1, 1], scope='logits')
            end_points['logits'] = logits
            return logits, end_points



if __name__ == '__main__':
    config = Config()
    input = tf.ones(shape=(1, 512, 512, 3), dtype=tf.float32)
    Unet(input, config)