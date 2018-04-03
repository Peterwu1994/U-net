#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-4-3 下午4:03
# @Author  : Yu-dong Wu
# @Site    : 
# @File    : Preprocessing.py
# @Software: PyCharm
# @profile :
import tensorflow as tf
import numpy as np

GT_COLOR = [0, 255, 0]
PRED_COLOR = [255, 0, 0]

def preprocess_image_and_label_Simple(image, label, size, is_training=True):
    """Preprocesses the image and label.

    Args:
      image: Input image.
      label: Ground truth annotation label.
      crop_height: The height value used to crop the image and label.
      crop_width: The width value used to crop the image and label.
      min_resize_value: Desired size of the smaller image side.
      max_resize_value: Maximum allowed size of the larger image side.
      resize_factor: Resized dimensions are multiple of factor plus one.
      min_scale_factor: Minimum scale factor value.
      max_scale_factor: Maximum scale factor value.
      scale_factor_step_size: The step size from min scale factor to max scale
        factor. The input is randomly scaled based on the value of
        (min_scale_factor, max_scale_factor, scale_factor_step_size).
      ignore_label: The label value which will be ignored for training and
        evaluation.
      is_training: If the preprocessing is used for training or not.
      model_variant: Model variant (string) for choosing how to mean-subtract the
        images. See feature_extractor.network_map for supported model variants.

    Returns:
      original_image: Original image (could be resized).
      processed_image: Preprocessed image.
      label: Preprocessed ground truth segmentation label.

    Raises:
      ValueError: Ground truth label not provided during training.
    """
    if is_training and label is None:
        raise ValueError('During training, label must be provided.')

    # Keep reference to original image.
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    if label is not None:
        label = tf.cast(label, tf.int32)

    processed_image = tf.image.resize_images(image, size=[size, size], method=tf.image.ResizeMethod.BILINEAR)
    label = tf.image.resize_images(label, size=[size, size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    if is_training:
        # Randomly left-right flip the image and label.
        processed_image, label, _ = flip_dim([processed_image, label], 0.5, dim=1)

    tf.summary.image('original_image', tf.expand_dims(image, 0))
    tf.summary.image('processed_image', tf.expand_dims(processed_image, 0))
    # label_visualize = tf.py_func(Visualize_label, [tf.expand_dims(label, 0), GT_COLOR], tf.uint8)
    label_visualize = tf.expand_dims(tf.cast(label, tf.uint8) * 255, 0)
    tf.summary.image('label', label_visualize)
    return processed_image, label


def flip_dim(tensor_list, prob=0.5, dim=1):
    """Randomly flips a dimension of the given tensor.

    The decision to randomly flip the `Tensors` is made together. In other words,
    all or none of the images pass in are flipped.

    Note that tf.random_flip_left_right and tf.random_flip_up_down isn't used so
    that we can control for the probability as well as ensure the same decision
    is applied across the images.

    Args:
      tensor_list: A list of `Tensors` with the same number of dimensions.
      prob: The probability of a left-right flip.
      dim: The dimension to flip, 0, 1, ..

    Returns:
      outputs: A list of the possibly flipped `Tensors` as well as an indicator
      `Tensor` at the end whose value is `True` if the inputs were flipped and
      `False` otherwise.

    Raises:
      ValueError: If dim is negative or greater than the dimension of a `Tensor`.
    """
    random_value = tf.random_uniform([])

    def flip():
        flipped = []
        for tensor in tensor_list:
            if dim < 0 or dim >= len(tensor.get_shape().as_list()):
                raise ValueError('dim must represent a valid dimension.')
            flipped.append(tf.reverse_v2(tensor, [dim]))
        return flipped

    is_flipped = tf.less_equal(random_value, prob)
    outputs = tf.cond(is_flipped, flip, lambda: tensor_list)
    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]
    outputs.append(is_flipped)

    return outputs


def Visualize_label(label, color):
    """
    visualize the label
    :param label: [batch_norm, height, width, 1]
    :param color:
    :return:
    """
    n, h, w, c = label.shape
    assert c == 1
    label = np.squeeze(label, 3)
    label = label[0,:,:]
    outputs = np.zeros((h, w, 3), dtype=np.uint8)
    outputs[np.where(label!=0), :] = color
    return outputs