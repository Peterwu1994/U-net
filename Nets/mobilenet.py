#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-4-4 上午8:53
# @Author  : Yu-dong Wu
# @Site    : 
# @File    : mobilenet.py
# @Software: PyCharm
# @profile :
import tensorflow as tf
from collections import namedtuple
from Config import Config
import functools
slim = tf.contrib.slim

Conv = namedtuple('Conv', ['kernel', 'stride', 'depth'])
DepthSepConv = namedtuple('DepthSepConv', ['kernel', 'stride', 'depth'])
_CONV_DEFS = [
    Conv(kernel=[3, 3], stride=2, depth=32),
    DepthSepConv(kernel=[3, 3], stride=1, depth=64),
    DepthSepConv(kernel=[3, 3], stride=2, depth=128),
    DepthSepConv(kernel=[3, 3], stride=1, depth=128),
    DepthSepConv(kernel=[3, 3], stride=2, depth=256),
    DepthSepConv(kernel=[3, 3], stride=1, depth=256),
    DepthSepConv(kernel=[3, 3], stride=2, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=2, depth=1024),
    DepthSepConv(kernel=[3, 3], stride=1, depth=1024)
]


def mobilenet_v1_base(inputs,
                      final_endpoint='Conv2d_13_pointwise',
                      min_depth=8,
                      depth_multiplier=1.0,
                      conv_defs=None,
                      output_stride=None,
                      scope=None):
    """Mobilenet v1.

    Constructs a Mobilenet v1 network from inputs to the given final endpoint.

    Args:
      inputs: a tensor of shape [batch_size, height, width, channels].
      final_endpoint: specifies the endpoint to construct the network up to. It
        can be one of ['Conv2d_0', 'Conv2d_1_pointwise', 'Conv2d_2_pointwise',
        'Conv2d_3_pointwise', 'Conv2d_4_pointwise', 'Conv2d_5'_pointwise,
        'Conv2d_6_pointwise', 'Conv2d_7_pointwise', 'Conv2d_8_pointwise',
        'Conv2d_9_pointwise', 'Conv2d_10_pointwise', 'Conv2d_11_pointwise',
        'Conv2d_12_pointwise', 'Conv2d_13_pointwise'].
      min_depth: Minimum depth value (number of channels) for all convolution ops.
        Enforced when depth_multiplier < 1, and not an active constraint when
        depth_multiplier >= 1.
      depth_multiplier: Float multiplier for the depth (number of channels)
        for all convolution ops. The value must be greater than zero. Typical
        usage will be to set this value in (0, 1) to reduce the number of
        parameters or computation cost of the model.
      conv_defs: A list of ConvDef namedtuples specifying the net architecture.
      output_stride: An integer that specifies the requested ratio of input to
        output spatial resolution. If not None, then we invoke atrous convolution
        if necessary to prevent the network from reducing the spatial resolution
        of the activation maps. Allowed values are 8 (accurate fully convolutional
        mode), 16 (fast fully convolutional mode), 32 (classification mode).
      scope: Optional variable_scope.

    Returns:
      tensor_out: output tensor corresponding to the final_endpoint.
      end_points: a set of activations for external use, for example summaries or
                  losses.

    Raises:
      ValueError: if final_endpoint is not set to one of the predefined values,
                  or depth_multiplier <= 0, or the target output_stride is not
                  allowed.
    """
    depth = lambda d: max(int(d * depth_multiplier), min_depth)
    end_points = {}

    # Used to find thinned depths for each layer.
    if depth_multiplier <= 0:
        raise ValueError('depth_multiplier is not greater than zero.')

    if conv_defs is None:
        conv_defs = _CONV_DEFS

    if output_stride is not None and output_stride not in [8, 16, 32]:
        raise ValueError('Only allowed output_stride values are 8, 16, 32.')

    with tf.variable_scope(scope, 'MobilenetV1', [inputs]):
        with slim.arg_scope([slim.conv2d, slim.separable_conv2d], padding='SAME'):
            # The current_stride variable keeps track of the output stride of the
            # activations, i.e., the running product of convolution strides up to the
            # current network layer. This allows us to invoke atrous convolution
            # whenever applying the next convolution would result in the activations
            # having output stride larger than the target output_stride.
            current_stride = 1

            # The atrous convolution rate parameter.
            rate = 1

            net = inputs
            for i, conv_def in enumerate(conv_defs):
                end_point_base = 'Conv2d_%d' % i

                if output_stride is not None and current_stride == output_stride:
                    # If we have reached the target output_stride, then we need to employ
                    # atrous convolution with stride=1 and multiply the atrous rate by the
                    # current unit's stride for use in subsequent layers.
                    layer_stride = 1
                    layer_rate = rate
                    rate *= conv_def.stride
                else:
                    layer_stride = conv_def.stride
                    layer_rate = 1
                    current_stride *= conv_def.stride

                if isinstance(conv_def, Conv):
                    end_point = end_point_base
                    net = slim.conv2d(net, depth(conv_def.depth), conv_def.kernel,
                                      stride=conv_def.stride,
                                      normalizer_fn=slim.batch_norm,
                                      scope=end_point)
                    end_points[end_point] = net
                    if end_point == final_endpoint:
                        return net, end_points

                elif isinstance(conv_def, DepthSepConv):
                    end_point = end_point_base + '_depthwise'

                    # By passing filters=None
                    # separable_conv2d produces only a depthwise convolution layer
                    net = slim.separable_conv2d(net, None, conv_def.kernel,
                                                depth_multiplier=1,
                                                stride=layer_stride,
                                                rate=layer_rate,
                                                normalizer_fn=slim.batch_norm,
                                                scope=end_point)

                    end_points[end_point] = net
                    if end_point == final_endpoint:
                        return net, end_points

                    end_point = end_point_base + '_pointwise'

                    net = slim.conv2d(net, depth(conv_def.depth), [1, 1],
                                      stride=1,
                                      normalizer_fn=slim.batch_norm,
                                      scope=end_point)

                    end_points[end_point] = net
                    if end_point == final_endpoint:
                        return net, end_points
                else:
                    raise ValueError('Unknown convolution type %s for layer %d'
                                     % (conv_def.ltype, i))
    raise ValueError('Unknown final endpoint %s' % final_endpoint)


def mobilenet_v1_arg_scope(is_training=True,
                           weight_decay=0.00004,
                           stddev=0.09,
                           regularize_depthwise=False):
    """Defines the default MobilenetV1 arg scope.

    Args:
      is_training: Whether or not we're training the model.
      weight_decay: The weight decay to use for regularizing the model.
      stddev: The standard deviation of the trunctated normal weight initializer.
      regularize_depthwise: Whether or not apply regularization on depthwise.

    Returns:
      An `arg_scope` to use for the mobilenet v1 model.
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
    if regularize_depthwise:
        depthwise_regularizer = regularizer
    else:
        depthwise_regularizer = None
    with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                        weights_initializer=weights_init,
                        activation_fn=tf.nn.relu6, normalizer_fn=slim.batch_norm):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            with slim.arg_scope([slim.conv2d], weights_regularizer=regularizer):
                with slim.arg_scope([slim.separable_conv2d],
                                    weights_regularizer=depthwise_regularizer) as sc:
                    return sc


def MobileNetSeg(inputs, config):
    argsc = mobilenet_v1_arg_scope(is_training=config.is_training, weight_decay=config.weight_decay)
    with slim.arg_scope(argsc):
        input_shape = inputs.get_shape().as_list()
        if len(input_shape) != 4:
            raise ValueError('Invalid input tensor rank, expected 4, was: %d' % len(input_shape))

        with tf.variable_scope('MobilenetV1', 'MobilenetV1', [inputs], reuse=False) as scope:
            with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=config.is_training):
                net, end_points = mobilenet_v1_base(inputs, scope='MobilenetV1')
                pass



if __name__ == '__main__':
    config = Config()
    input = tf.ones(shape=(1, 512, 512, 3), dtype=tf.float32)
    MobileNetSeg(input, config)
