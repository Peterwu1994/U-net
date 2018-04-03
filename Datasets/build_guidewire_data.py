#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-3-29 下午3:59
# @Author  : Yu-dong Wu
# @Site    : 
# @File    : build_guidewire_data.py
# @Software: PyCharm
# @profile :
import glob
import math
import os.path
import sys
import build_data
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tqdm import tqdm

sys.path.append('/home/wuyudong/Project/scripts/single_files/')
from UtilsSelf import FindAllFiles


def _convert_dataset(img_dir, label_dir, record_path):
    """
    write all samples in one tfrecord
    :param img_dir:
    :return:
    """
    all_image_paths = FindAllFiles(img_dir, ['png'])

    image_reader = build_data.ImageReader('jpeg', channels=3)
    label_reader = build_data.ImageReader('png', channels=1)

    if not os.path.isdir(os.path.dirname(record_path)):
        os.makedirs(os.path.dirname(record_path))
        print('mkdir: %s' % os.path.dirname(record_path))

    with tf.python_io.TFRecordWriter(record_path) as tfrecord_writer:
        for img_path in tqdm(all_image_paths):
            # get label path
            label_path = img_path.replace(os.path.abspath(img_dir), os.path.abspath(label_dir))
            if not os.path.isfile(label_path):
                raise Exception('cannot find label image %s' % label_path)
            image_data = tf.gfile.FastGFile(img_path, 'rb').read()
            height, width = image_reader.read_image_dims(image_data)

            seg_data = tf.gfile.FastGFile(label_path, 'rb').read()
            seg_height, seg_width = label_reader.read_image_dims(seg_data)

            if height != seg_height or width != seg_width:
                raise RuntimeError('Shape mismatched between image and label.')

            # Convert to tf example.
            example = build_data.image_seg_to_tfexample(image_data, str.encode(img_path), height, width, seg_data)
            tfrecord_writer.write(example.SerializeToString())


if __name__ == '__main__':
    _convert_dataset(img_dir='/home/wuyudong/Project/ImageData/GuideWire/Image/TrainSequences',
                     label_dir='/home/wuyudong/Project/ImageData/GuideWire/Label/TrainSequences_binary',
                     record_path='/home/wuyudong/Project/ImageData/GuideWire/Image/Train.tfrecord')