#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-4-4 上午8:53
# @Author  : Yu-dong Wu
# @Site    : 
# @File    : mobilenet_v1.py
# @Software: PyCharm
# @profile :
import tensorflow as tf
from collections import namedtuple
# from Config import co
import functools
slim = tf.contrib.slim

Conv = namedtuple('Conv', ['kernel', 'stride', 'depth'])
DepthSepConv = namedtuple('DepthSepConv', ['kernel', 'stride', 'depth'])
TranseConv = namedtuple('TranseConv', ['kernel', 'stride', 'depth'])
Concat = namedtuple('Concat', ['src'])

_DECODER = [
    [TranseConv(kernel=[3, 3], stride=2, depth=512),
     Concat(src='Conv2d_11_pointwise'),
     Conv(kernel=[3, 3], stride=1, depth=512)],

    [TranseConv(kernel=[3, 3], stride=2, depth=256),
     Concat(src='Conv2d_5_pointwise'),
     Conv(kernel=[3, 3], stride=1, depth=256)],

    [TranseConv(kernel=[3, 3], stride=2, depth=128),
     Concat(src='Conv2d_3_pointwise'),
     Conv(kernel=[3, 3], stride=1, depth=128)],

    [TranseConv(kernel=[3, 3], stride=2, depth=64),
     Concat(src='Conv2d_1_pointwise'),
     Conv(kernel=[3, 3], stride=1, depth=64)],

    [TranseConv(kernel=[3, 3], stride=2, depth=32),
     Concat(src='Conv2d_Aux'),
     Conv(kernel=[3, 3], stride=1, depth=32)],

]


# _DECODER_FOR_ATROUS = [
#     [TranseConv(kernel=[3, 3], stride=2, depth=128),
#      Concat(src='Conv2d_3_pointwise'),
#      Conv(kernel=[3, 3], stride=1, depth=128)],
#
#     [TranseConv(kernel=[3, 3], stride=2, depth=64),
#      Concat(src='Conv2d_1_pointwise'),
#      Conv(kernel=[3, 3], stride=1, depth=64)],
#
#     [TranseConv(kernel=[3, 3], stride=2, depth=32),
#      Concat(src='Conv2d_Aux'),
#      Conv(kernel=[3, 3], stride=1, depth=32)],
#
# ]


_OutputStride2DecoderBeg = {32: 0,
                            16: 1,
                            8: 2,}

_OutputStride2Endpoint = {32: 'Conv2d_13_pointwise',
                          16: 'Conv2d_11_pointwise',
                          8: 'Conv2d_5_pointwise',}

_CONV_DEFS = [
    Conv(kernel=[3, 3], stride=1, depth=16),  # TODO not right
    Conv(kernel=[3, 3], stride=1, depth=16),
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
                      final_endpoint='',
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
    strides2feat = {}

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
                    strides2feat['stride%d' % current_stride] = net
                    if end_point == final_endpoint:
                        return net, end_points, strides2feat

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

                    end_point = end_point_base + '_pointwise'
                    d = depth(conv_def.depth)
                    net = slim.conv2d(net, depth(conv_def.depth), [1, 1],
                                      stride=1,
                                      normalizer_fn=slim.batch_norm,
                                      scope=end_point)

                    end_points[end_point] = net
                    strides2feat['stride%d' % current_stride] = net

                    if end_point == final_endpoint:
                        return net, end_points, strides2feat
                else:
                    raise ValueError('Unknown convolution type %s for layer %d'
                                     % (conv_def.ltype, i))

            return net, end_points, strides2feat


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
            with slim.arg_scope([slim.conv2d], weights_regularizer=regularizer, padding='SAME'):
                with slim.arg_scope([slim.separable_conv2d],
                                    weights_regularizer=depthwise_regularizer) as sc:
                    return sc


def MobileNetSeg(inputs, config):
    """

    :param inputs:
    :param config: max_stride need to be set: typical values [32, 16, 8, 4]
    :return:
    """
    print(config.max_stride)
    argsc = mobilenet_v1_arg_scope(is_training=config.is_training, weight_decay=config.weight_decay)
    with slim.arg_scope(argsc):
        input_shape = inputs.get_shape().as_list()
        if len(input_shape) != 4:
            raise ValueError('Invalid input tensor rank, expected 4, was: %d' % len(input_shape))

        if config.max_stride not in _OutputStride2Endpoint.keys():
            raise ValueError('Invalid max_stride %d not in [32, 16, 8]' % config.max_stride)

        with tf.variable_scope('MobilenetV1', 'MobilenetV1', [inputs], reuse=False) as scope:
            # encoder
            with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=config.is_training):
                net, end_points = mobilenet_v1_base(inputs, final_endpoint=_OutputStride2Endpoint[config.max_stride],
                                                    scope=scope)
                conv_aux = slim.conv2d(inputs, 32, [3, 3], stride=1, scope='Conv2d_Aux')
                end_points['Conv2d_Aux'] = conv_aux
                if config.max_stride == 4:
                    net = slim.repeat(net, 4, slim.conv2d, 128, [3, 3], scope='conv_repeat')
                elif config.max_stride == 8:
                    net = slim.repeat(net, 4, slim.conv2d, 256, [3, 3], scope='conv_repeat')
            # decoder
            with slim.arg_scope([slim.conv2d_transpose, slim.conv2d], activation_fn=tf.nn.relu,
                            weights_regularizer=slim.l2_regularizer(config.weight_decay),
                            biases_initializer=tf.zeros_initializer(), normalizer_fn=slim.batch_norm,
                            padding='SAME',):
                beg = _OutputStride2DecoderBeg[config.max_stride]
                decoder = _DECODER[beg:]
                for i, conv_unit in enumerate(decoder):
                    for conv_def in conv_unit:
                        if isinstance(conv_def, TranseConv):
                            net = slim.conv2d_transpose(net, conv_def.depth, conv_def.kernel, stride=conv_def.stride,
                                                        scope='Deconv%d'%i)
                            end_points['Deconv%d' % i] = net

                        elif isinstance(conv_def, Concat):
                            net = tf.concat([net, end_points[conv_def.src]], axis=3)
                        elif isinstance(conv_def, Conv):
                            net = slim.conv2d(net, conv_def.depth, conv_def.kernel, stride=conv_def.stride,
                                              scope='Conv%d'%(i+14))
                        else:
                            raise Exception('not recongize conv_def')

                logits = slim.conv2d(net, 2, [3, 3], stride=1, normalizer_fn=None, scope='logits')
                end_points['logits'] = logits
                return logits, end_points


def MobileNetSegAtrous(inputs, config):
    argsc = mobilenet_v1_arg_scope(is_training=config.is_training, weight_decay=config.weight_decay)
    with slim.arg_scope(argsc):
        input_shape = inputs.get_shape().as_list()
        if len(input_shape) != 4:
            raise ValueError('Invalid input tensor rank, expected 4, was: %d' % len(input_shape))

        if config.max_stride not in _OutputStride2Endpoint.keys():
            raise ValueError('Invalid max_stride %d not in [32, 16, 8, 4]' % config.max_stride)

        with tf.variable_scope('MobilenetV1', 'MobilenetV1', [inputs], reuse=False) as scope:
            # encoder
            with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=config.is_training):
                net, end_points = mobilenet_v1_base(inputs, final_endpoint='Conv2d_13_pointwise',
                                                    scope=scope, output_stride=config.max_stride)
                conv_aux = slim.conv2d(inputs, 32, [3, 3], stride=1, scope='Conv2d_Aux')
                end_points['Conv2d_Aux'] = conv_aux
            # decoder
            with slim.arg_scope([slim.conv2d_transpose, slim.conv2d], activation_fn=tf.nn.relu,
                                weights_regularizer=slim.l2_regularizer(config.weight_decay),
                                biases_initializer=tf.zeros_initializer(), normalizer_fn=slim.batch_norm,
                                padding='SAME', ):
                beg = _OutputStride2DecoderBeg[config.max_stride]
                decoder = _DECODER[beg:]
                for i, conv_unit in enumerate(decoder):
                    for conv_def in conv_unit:
                        if isinstance(conv_def, TranseConv):
                            net = slim.conv2d_transpose(net, conv_def.depth, conv_def.kernel, stride=conv_def.stride,
                                                        scope='Deconv%d'%i)
                            end_points['Deconv%d' % i] = net

                        elif isinstance(conv_def, Concat):
                            net = tf.concat([net, end_points[conv_def.src]], axis=3)
                        elif isinstance(conv_def, Conv):
                            net = slim.conv2d(net, conv_def.depth, conv_def.kernel, stride=conv_def.stride,
                                              scope='Conv%d'%(i+14))
                        else:
                            raise Exception('not recongize conv_def')

                logits = slim.conv2d(net, 2, [3, 3], stride=1, normalizer_fn=None, scope='logits')
                end_points['logits'] = logits
                return logits, end_points

#TODO implement a decoder


if __name__ == '__main__':
    # config = Config(max_stride=16)
    # input = tf.ones(shape=(1, 512, 512, 3), dtype=tf.float32)
    # MobileNetSegAtrous(input, config)
    pass