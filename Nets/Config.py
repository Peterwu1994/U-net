#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-4-3 上午11:34
# @Author  : Yu-dong Wu
# @Site    : 
# @File    : Config.py
# @Software: PyCharm
# @profile :

class Config():
    def __init__(self, weight_decay=0.0004, is_training=True):
        self.weight_decay = weight_decay
        self.is_training = is_training