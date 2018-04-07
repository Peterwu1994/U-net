#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-4-4 下午2:23
# @Author  : Yu-dong Wu
# @Site    : 
# @File    : losses.py
# @Software: PyCharm
# @profile :
import tensorflow as tf
slim = tf.contrib.slim


def dice_loss(logits, labels, epsilon=1e-7, ratio=1):
    '''
    calculate the dice loss: DL = -2 * sum(B * Bp) / (sum(B) + sum(Bp))
    :param logits:[b, h, w, 2]
    :param labels:[b, h, w]
    :return:
    '''
    logits = tf.nn.softmax(logits)
    logits = logits[:, 1]
    labels = tf.cast(labels, dtype=tf.float32)
    dl = -2.0 * (tf.reduce_sum(labels * logits) + epsilon) / (tf.reduce_sum(logits) + tf.reduce_sum(labels))
    return dl * ratio


def weighted_softmax_loss(logits, labels, weight_ratio=(1, 400)):
    """

    :param logits:
    :param labels: one-hot labels
    :param weight_ratio:
    :return:
    """
    labels = slim.one_hot_encoding(labels, 2)
    logits = tf.reshape(logits, shape=[-1, logits.get_shape()[-1]])
    labels = tf.reshape(labels, shape=[-1, labels.get_shape()[-1]])
    class_weights = tf.constant(weight_ratio, dtype=tf.float32)
    weights = tf.reduce_sum(class_weights * labels, axis=1)
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    return tf.reduce_mean(loss * weights)