#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-4-3 上午11:34
# @Author  : Yu-dong Wu
# @Site    : 
# @File    : Config.py
# @Software: PyCharm
# @profile : config
from easydict import EasyDict as edict
from Datasets.dataset_common import count_samples_in

config = edict()
config.is_training = True
# config for network
config.Network = edict()
config.Network.encoder_backbone = 'MobilenetV1' # MobilenetV1 Ori_Unet
config.Network.max_stride = 16

# config for dataset
config.Dataset = edict()
config.Dataset.seg_num_class = 2
config.Dataset.dataset_dir = '/home/wuyudong/Project/ImageData/GuideWire/Image_debug/Train.tfrecord'
config.Dataset.num_samples = count_samples_in(config.Dataset.dataset_dir)
# config for train
config.Train = edict()
config.Train.weight_decay = 0.00004
config.Train.batch_size = 2
config.Train.image_size = [512, 512]
config.Train.seg_loss_func = 'dice'  # weighted_softmax dice
# slover
config.Train.learning_rate_decay_type = 'fixed'  # 'fixed', 'polynomial'
config.Train.learning_rate = 0.1
config.Train.learning_rate_decay_factor = 0.94
config.Train.num_epochs_per_decay = 2
config.Train.end_learning_rate = 0.0001
config.Train.optimizer = 'sgd'  # 'adadelta', 'adagrad', 'adam', 'ftrl', 'momentum', 'rmsprop', 'sgd'
config.Train.opt_epsilon = None

# adadelta
config.Train.adadelta_rho = 0.95
# adagrad
config.Train.adagrad_initial_accumulator_value = 0.1
# adam
config.Train.adam_beta1 = 0.9
config.Train.adam_beta2 = 0.999
# ftrl
config.Train.ftrl_learning_rate_power = -0.5
config.Train.ftrl_initial_accumulator_value = 0.1
config.Train.ftrl_l1 = 0.0
config.Train.ftrl_l2 = 0.0
# momentum
config.Train.momentum = 0.9
# rmsprop
config.Train.rmsprop_decay = 0.9
config.Train.rmsprop_momentum = 0.9

config.Train.moving_average_decay = None
config.Train.train_log_dir = '/raid/wuyudong/Models/Unet/Seg_stride16'
config.Train.max_number_of_steps = int(8 * config.Dataset.num_samples / config.Train.batch_size)
config.Train.log_every_n_steps = 10
config.Train.save_summaries_secs = 60
config.Train.save_interval_secs = 1200

# config.Train.checkpoint_path = '/raid/wuyudong/Models/PreTrainedModels/mobile_1/mobilenet_v1_1.0_224.ckpt'  # 'The path to a checkpoint from which to fine-tune.'
config.Train.checkpoint_path = None
config.Train.checkpoint_exclude_scopes = None
config.Train.trainable_scopes = None

