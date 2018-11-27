#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-4-9 上午10:02
# @Author  : Yu-dong Wu
# @Site    : 
# @File    : Inference.py
# @Software: PyCharm
# @profile :
import os
import sys

import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from Datasets.SegmentationDataset import get_dataset
from Nets.mobilenet_v1 import MobileNetSeg
from Preprocessing.Preprocessing import preprocess_image_and_label_Simple
from Config.Config import Config
from Metric import *
import pickle
from tqdm import tqdm
slim = tf.contrib.slim

# ckpt_dir = '/raid/wuyudong/Models/DeeplabGuidewire/Mobilenet/Train/Atrous16'
checkpoint_file = '/raid/wuyudong/Models/DeeplabGuidewire/Mobilenet/Train/Stride32NewTrainData_1/checkpoint'
# eval_dir = '/raid/wuyudong/Models/DeeplabGuidewire/Mobilenet/Eval/Stride16'
dataset_dir = '/home/wuyudong/Project/ImageData/GuideWire/Image/NewTest.tfrecord'
rst_file = '/raid/wuyudong/Models/DeeplabGuidewire/Mobilenet/Inference/Stride32NewTrainData_1.pickle'

batch_size = 16
eval_image_size = 512
num_samples = 474
max_stride = 32


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
        # batch_queue = slim.prefetch_queue.prefetch_queue(
        #     [images, labels, image_names], capacity=2 * 1)
        # images, labels, image_names = batch_queue.dequeue()
        network_config = Config(is_training=False, max_stride=max_stride)
        logits, endpoints = network_fn(images, network_config)
        pred = tf.argmax(logits, axis=-1)
        pred = tf.expand_dims(pred, -1)

        saver = tf.train.Saver(tf.global_variables())

        def SessRun(ckpt_path):
            gpu_options = tf.GPUOptions(allow_growth=True)
            num_loops = int(dataset.num_samples / batch_size)
            with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False)) as sess:
                saver.restore(sess, ckpt_path)
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess, coord)
                distances_l_r = []
                distances_r_l = []
                for i in tqdm(range(num_loops)):
                    ls, ps, ori_im = sess.run([labels, pred, images])
                    ls = ls.astype(np.uint8)
                    ps = ps.astype(np.uint8)
                    ls = ls * 255
                    ps = ps * 255
                    # io.imsave(os.path.join('/home/wuyudong/Project/ImageData/Temp/I', '%d.png' % i), np.squeeze(ori_im[0]))
                    # io.imsave(os.path.join('/home/wuyudong/Project/ImageData/Temp/P', '%d.png' % i), np.squeeze(ps[0]))
                    # io.imsave(os.path.join('/home/wuyudong/Project/ImageData/Temp/L', '%d.png' % i), np.squeeze(ls[0]))
                    # metric
                    dist_l_r, dist_r_l = DistancesFromLeftToRightsMultis(ps, ls, skeletonize=False, closeOp=False)
                    distances_l_r += dist_l_r
                    distances_r_l += dist_r_l

                coord.request_stop()
                coord.join(threads)
                return DistancesToMetrics(distances_l_r, distances_r_l)

        if not os.path.isdir(os.path.dirname(rst_file)):
            os.makedirs(os.path.dirname(rst_file))

        with open(checkpoint_file, 'r') as ckpt_file:
            ckpt_file.readline()

            step2metrics = {}
            for line in ckpt_file.readlines():
                ckpt_path, step = ParseCheckpoint(line)
                if step <= 30000:
                    continue
                rst = SessRun(ckpt_path)
                step2metrics[step] = rst
                print('%d: mean:%f\tstd:%f\tmedian:%f\t\tfalse:%f\tmissing:%f\tF1:%f' % (step, rst.dist_mean,
                                                                                         rst.dist_std, rst.dist_median,
                                                                                         rst.false, rst.missing, rst.F1) )
                sys.stdout.flush()
            rst_f = open(rst_file, 'wb')
            steps = list(step2metrics.keys())
            steps.sort()
            for step in steps:
                rst = step2metrics[step]
                print('%d: mean:%f\tstd:%f\tmedian:%f\t\tfalse:%f\tmissing:%f\tF1:%f' % (step, rst.dist_mean,
                                                                                         rst.dist_std, rst.dist_median,
                                                                                         rst.false, rst.missing,
                                                                                         rst.F1))
            pickle.dump(step2metrics, rst_f)
            print('pickle dump rst to %s' % rst_file)




def ParseCheckpoint(checkpoint_line):
    _, ckpt_path, _ = checkpoint_line.split('"')
    step = int(ckpt_path.rsplit('-')[1])
    return ckpt_path, step



if __name__ == '__main__':
    tf.app.run()
    # f = open(rst_file, 'rb')
    # step2metrics = pickle.load(f)
    # steps = list(step2metrics.keys())
    # steps.sort()
    # for step in steps:
    #     rst = step2metrics[step]
    #     print('%d: mean:%f\tstd:%f\tmedian:%f\t\tfalse:%f\tmissing:%f\tF1:%f' % (step, rst.dist_mean,
    #                                                                              rst.dist_std, rst.dist_median,
    #                                                                              rst.false, rst.missing,
    #                                                                              rst.F1))