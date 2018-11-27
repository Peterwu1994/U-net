#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-4-7 下午11:10
# @Author  : Yu-dong Wu
# @Site    : 
# @File    : eval.py
# @Software: PyCharm
# @profile :
import os

import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from Datasets.SegmentationDataset import get_dataset
from Preprocessing.Preprocessing import preprocess_image_and_label_Simple
from Config.Config import config
from unet import unet
import math
from common import print_config
from deployment import model_deploy
slim = tf.contrib.slim
config.is_training = False
ckpt_dir = '/raid/wuyudong/Models/Unet/Seg_stride16'
eval_dir = '/raid/wuyudong/Models/Unet/Seg_eval'
dataset_dir = '/home/wuyudong/Project/ImageData/GuideWire/Image_debug/Test.tfrecord'

batch_size = 1
eval_image_size = 512
num_samples = 360
max_stride = config.Network.max_stride

network_fn = unet
preprocess_fn = preprocess_image_and_label_Simple

def main(_):
    global config
    print_config(config)
    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():
        deploy_config = model_deploy.DeploymentConfig()
        # Create global_step
        with tf.device(deploy_config.variables_device()):
            global_step = tf.contrib.training.get_or_create_eval_step()
        dataset = get_dataset(dataset_name='guidewire', dataset_dir=dataset_dir)
        with tf.device(deploy_config.inputs_device()):
            provider = slim.dataset_data_provider.DatasetDataProvider(
                dataset,
                shuffle=False,
                common_queue_capacity=20 * batch_size,
                common_queue_min=10 * batch_size)
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
            # network_config = Config(is_training=False, max_stride=max_stride)
            network_fn = unet
            logits, endpoints = network_fn(images, config)
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

        if tf.gfile.IsDirectory(ckpt_dir):
            checkpoint_path = tf.train.latest_checkpoint(ckpt_dir)
        else:
            checkpoint_path = ckpt_dir

        tf.logging.info('Evaluating %s' % checkpoint_path)

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0, allow_growth=True)
        sess_config = tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False)

        # slim.evaluation.evaluation_loop(
        #     master='',
        #     checkpoint_dir=ckpt_dir,
        #     logdir=eval_dir,
        #     num_evals=1,
        #     eval_op=list(names_to_updates.values()),
        #     variables_to_restore=variables_to_restore,
        #     eval_interval_secs=20,
        #     summary_op=tf.summary.merge_all(),
        #     session_config=sess_config)

        slim.evaluation.evaluate_once(
            master='',
            checkpoint_path='/raid/wuyudong/Models/Unet/Seg_stride16/model.ckpt-5840',
            logdir=eval_dir,
            variables_to_restore=variables_to_restore,
            eval_op=list(names_to_updates.values()),
            summary_op=tf.summary.merge_all(),
            session_config=sess_config
        )

if __name__ == '__main__':
    tf.app.run()