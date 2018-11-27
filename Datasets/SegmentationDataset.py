#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-4-3 上午11:00
# @Author  : Yu-dong Wu
# @Site    : 
# @File    : SegmentationDataset.py
# @Software: PyCharm
# @profile : instance dataset from tfrecord

import collections
import os
import tensorflow as tf
from Config.Config import config


slim = tf.contrib.slim

dataset = slim.dataset

tfexample_decoder = slim.tfexample_decoder


_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying height and width.',
    'labels_class': ('A semantic segmentation label whose size matches image.'
                     'Its values range from 0 (background) to num_classes.'),
}

# Named tuple to describe the dataset properties.
DatasetDescriptor = collections.namedtuple(
    'DatasetDescriptor',
    [
     'num_classes',   # Number of semantic classes.
     'ignore_label',  # Ignore label value.
     ]
)

_GUIDEWIRE_SEG_INFORMATION = DatasetDescriptor(
    num_classes=config.Dataset.seg_num_class,
    ignore_label=255,
)


_DATASETS_INFORMATION = {
    'guidewire': _GUIDEWIRE_SEG_INFORMATION,
}

keys_to_features = {
    'image/encoded': tf.FixedLenFeature(
        (), tf.string, default_value=''),
    'image/filename': tf.FixedLenFeature(
        (), tf.string, default_value=''),
    'image/format': tf.FixedLenFeature(
        (), tf.string, default_value='jpeg'),
    'image/height': tf.FixedLenFeature(
        (), tf.int64, default_value=0),
    'image/width': tf.FixedLenFeature(
        (), tf.int64, default_value=0),
    'image/segmentation/class/encoded': tf.FixedLenFeature(
        (), tf.string, default_value=''),
    'image/segmentation/class/format': tf.FixedLenFeature(
        (), tf.string, default_value='png'),
}

def get_files_size(files):
    """
    return sum of bytesizes of each file in files
    :param files:
    :return:
    """
    bytesize = 0
    for f in files:
        st = os.stat(f)
        bytesize += st
    return bytesize


def get_dataset(dataset_name, dataset_dir):
    """Gets an instance of slim Dataset.

    Args:
      dataset_dir: The directory of the dataset sources.

    Returns:
      An instance of slim Dataset.

    Raises:
      ValueError: if the dataset_name or split_name is not recognized.
    """
    # Prepare the variables for different datasets.
    global config
    if os.path.isdir(dataset_dir):
        tfrecords = os.listdir(dataset_dir)
        tfrecords = [i for i in tfrecords if i.endswith('tfrecord')]
    elif os.path.isfile(dataset_dir):
        tfrecords = [dataset_dir]
    else:
        raise Exception('invalid path: %s' % dataset_dir)

    print('%d records: %s in %s' %
          (len(tfrecords), str([os.path.basename(i) for i in tfrecords]), os.path.dirname(tfrecords[0])))
    # Specify how the TF-Examples are decoded.
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature(
            (), tf.string, default_value=''),
        'image/filename': tf.FixedLenFeature(
            (), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature(
            (), tf.string, default_value='jpeg'),
        'image/height': tf.FixedLenFeature(
            (), tf.int64, default_value=0),
        'image/width': tf.FixedLenFeature(
            (), tf.int64, default_value=0),
        'image/segmentation/class/encoded': tf.FixedLenFeature(
            (), tf.string, default_value=''),
        'image/segmentation/class/format': tf.FixedLenFeature(
            (), tf.string, default_value='png'),
    }
    items_to_handlers = {
        'image': tfexample_decoder.Image(
            image_key='image/encoded',
            format_key='image/format',
            channels=3),
        'image_name': tfexample_decoder.Tensor('image/filename'),
        'height': tfexample_decoder.Tensor('image/height'),
        'width': tfexample_decoder.Tensor('image/width'),
        'labels_class': tfexample_decoder.Image(
            image_key='image/segmentation/class/encoded',
            format_key='image/segmentation/class/format',
            channels=1),
    }

    decoder = tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)

    return dataset.Dataset(
        data_sources=tfrecords,
        reader=tf.TFRecordReader,
        decoder=decoder,
        num_samples=config.Dataset.num_samples,
        items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
        name=dataset_name,
        multi_label=True)



def vis_tfrecord(tfrecord_path, save_dir=None):
    """
    vis samples in tfrecord to verify
    tfrecord_path: paths to tfrecord, can be a list or single file
    save_dir: dir to save the visualized results
    :return:
    """
    import numpy as np
    import cv2
    # if isinstance(tfrecord_path, list) or isinstance(tfrecord_path, tuple):
    #     pass
    # else:
    #     tfrecord_path = [tfrecord_path]
    #
    # with tf.Session() as sess:
    #     filename_queue = tf.train.string_input_producer(tfrecord_path, num_epochs=1)
    #     reader = tf.TFRecordReader()
    #     _, serialized_example = reader.read(queue=filename_queue)
    #
    #     features = tf.parse_single_example(serialized_example, keys_to_features)
    #     image = tf.decode_raw(features['image/encoded'], tf.string)
    #     image = tf.image.decode_image(image, channels=3)
    #     label = tf.decode_raw(features['image/segmentation/class/encoded'], tf.string)
    #     label = tf.image.decode_image(label, channels=1)
    #     # height = tf.cast(features['image/height'], tf.int64)
    #     # width = tf.cast(features['image/width'], tf.int64)
    #     # image = tf.reshape(image, tf.stack([height, width, 3]))
    #     # label = tf.reshape(label, tf.stack([height, width, 1]))
    #     # image, label = tf.train.batch([image, label], batch_size=1,)
    #     init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    #     sess.run(init_op)
    #
    #     coord = tf.train.Coordinator()
    #     threads = tf.train.start_queue_runners(coord=coord)
    #     for i in range(10):
    #         img, l = sess.run([image, label])
    #         img = img.astype(np.uint8)
    #         print(img.shape)
    #         l = l.astype(np.uint8)
    #         print(l.shape)
    #         # cv2.imshow('img', img)
    #         # cv2.imshow('label', l)
    #         # cv2.waitKey()
    #
    #     coord.request_stop()
    #     coord.join(threads)
    #     sess.close()
    with tf.Session() as sess:
        dataset = get_dataset(dataset_name='guidewire', dataset_dir=config.Dataset.dataset_dir)
        provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset,
            num_readers=4,
            common_queue_capacity=20,
            common_queue_min=10)
        [image, label] = provider.get(['image', 'labels_class'])

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(10):
            img, l = sess.run([image, label])
            img = img.astype(np.uint8)
            l = l.astype(np.uint8)
            l[np.where(l !=  0)] = 255
            cv2.imshow('img', img)
            cv2.imshow('label', l)
            cv2.waitKey()

        coord.request_stop()
        coord.join(threads)
        sess.close()



if __name__ == '__main__':
    pass
    # vis_tfrecord('/home/wuyudong/Project/ImageData/GuideWire/Image_debug/Test.tfrecord')
