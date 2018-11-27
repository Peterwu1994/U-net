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
from tensorflow.python.ops import control_flow_ops

GT_COLOR = [0, 255, 0]
PRED_COLOR = [255, 0, 0]


def preprocess_image_and_label_Simple(image, label, size, is_training=True, num_classes=2):
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
    if isinstance(size, int):
        size = [size, size]
    assert isinstance(size, list) or isinstance(size, tuple), 'size need to be list or tuple, now is %s' % type(size)
    image = tf.image.resize_images(image, size=size, method=tf.image.ResizeMethod.BILINEAR)
    label = tf.image.resize_images(label, size=size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    if num_classes == 2:
        label = binary_label(label)

    if is_training:
        # Randomly left-right flip the image and label.
        image, label, _ = flip_dim([image, label], 0.5, dim=1)

    return image, label


def binary_label(label):
    """
    convert label to binary for binary classification
    :param label:
    :param num_classes:
    :return:
    """
    label = tf.where(tf.equal(label, 0), tf.zeros_like(label), tf.ones_like(label))
    return label



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


def apply_with_random_selector(x, func, num_cases):
  """Computes func(x, sel), with sel sampled from [0...num_cases-1].

  Args:
    x: input Tensor.
    func: Python function to apply.
    num_cases: Python int32, number of cases to sample sel from.

  Returns:
    The result of func(x, sel), where func receives the value of the
    selector as a python integer, but sel is sampled dynamically.
  """
  sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)
  # Pass the real x only to one of the func calls.
  return control_flow_ops.merge([
      func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
      for case in range(num_cases)])[0]


def distort_color(image, color_ordering=0, fast_mode=True, scope=None):
  """Distort the color of a Tensor image.

  Each color distortion is non-commutative and thus ordering of the color ops
  matters. Ideally we would randomly permute the ordering of the color ops.
  Rather then adding that level of complication, we select a distinct ordering
  of color ops for each preprocessing thread.

  Args:
    image: 3-D Tensor containing single image in [0, 1].
    color_ordering: Python int, a type of distortion (valid values: 0-3).
    fast_mode: Avoids slower ops (random_hue and random_contrast)
    scope: Optional scope for name_scope.
  Returns:
    3-D Tensor color-distorted image on range [0, 1]
  Raises:
    ValueError: if color_ordering not in [0, 3]
  """
  with tf.name_scope(scope, 'distort_color', [image]):
    if fast_mode:
      if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      else:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
    else:
      if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
      elif color_ordering == 1:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
      elif color_ordering == 2:
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      elif color_ordering == 3:
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
      else:
        raise ValueError('color_ordering must be in [0, 3]')

    # The random_* ops do not necessarily clamp.
    return tf.clip_by_value(image, 0.0, 1.0)


def preprocess_image_and_label(image, label, size, is_training=True):
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

    image = tf.image.resize_images(image, size=[size, size], method=tf.image.ResizeMethod.BILINEAR)
    label = tf.image.resize_images(label, size=[size, size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    if is_training:
        # Randomly left-right flip the image and label.
        image, label, _ = flip_dim([image, label], 0.5, dim=1)

        image = apply_with_random_selector(
            image,
            lambda x, ordering: distort_color(x, ordering, fast_mode=False),
            num_cases=4)

    return image, label