#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-4-7 下午11:10
# @Author  : Yu-dong Wu
# @Site    : 
# @File    : eval.py
# @Software: PyCharm
# @profile :
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from Datasets.SegmentationDataset import get_dataset
from Nets.unet import Unet
from Nets.mobilenet import MobileNetSegAtrous, MobileNetSeg
from Preprocessing.Preprocessing import preprocess_image_and_label_Simple, Visualize_label, PRED_COLOR
from Nets.Config import Config
from losses import weighted_softmax_loss, dice_loss
import math
slim = tf.contrib.slim

ckpt_dir = '/raid/wuyudong/Models/DeeplabGuidewire/Mobilenet/Train/Stride16'
eval_dir = '/raid/wuyudong/Models/DeeplabGuidewire/Mobilenet/Eval/Stride16'
dataset_dir = '/home/wuyudong/Project/ImageData/GuideWire/Image/Test.tfrecord'

batch_size = 4
eval_image_size = 512
num_samples = 360
max_stride = 16

network_fn = MobileNetSeg
preprocess_fn = preprocess_image_and_label_Simple

def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():
        tf_global_step = slim.get_or_create_global_step()
        dataset = get_dataset(dataset_name='guidewire', split_name='val', dataset_dir=dataset_dir)

        provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset,
            shuffle=False,
            common_queue_capacity=2 * batch_size,
            common_queue_min=batch_size)
        [image, label] = provider.get(['image', 'labels_class'])
        image, label = preprocess_fn(image, label, eval_image_size, is_training=False)
        images, labels = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=4,
            capacity=5 * batch_size)
        batch_queue = slim.prefetch_queue.prefetch_queue(
            [images, labels], capacity=2 * 1)
        images, labels = batch_queue.dequeue()
        network_config = Config(is_training=False, max_stride=max_stride)
        logits, endpoints = network_fn(images, network_config)
        variables_to_restore = slim.get_variables_to_restore()

        pred = tf.argmax(logits, axis=-1)
        pred = tf.expand_dims(pred, -1)
        label_visualize = tf.cast(pred, tf.uint8) * 255
        tf.summary.image('pred', label_visualize)
        tf.summary.image('images', images)
        tf.summary.image('labels', tf.cast(labels, tf.uint8)*255)


        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
            'veal/mean_absolute': slim.metrics.streaming_mean_absolute_error(tf.cast(pred, tf.int64),
                                                                             tf.cast(labels,tf.int64))
        })

        num_batches = math.ceil(num_samples / float(batch_size))

        if tf.gfile.IsDirectory(ckpt_dir):
            checkpoint_path = tf.train.latest_checkpoint(ckpt_dir)
        else:
            checkpoint_path = ckpt_dir

        tf.logging.info('Evaluating %s' % checkpoint_path)

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0, allow_growth=True)
        config = tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False)

        slim.evaluation.evaluation_loop(
            master='',
            checkpoint_dir=ckpt_dir,
            logdir=eval_dir,
            num_evals=num_batches,
            eval_op=list(names_to_updates.values()),
            variables_to_restore=variables_to_restore,
            eval_interval_secs=200,
            summary_op=tf.summary.merge_all(),
            session_config=config)

if __name__ == '__main__':
    tf.app.run()