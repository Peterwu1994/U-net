#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-11-26 下午7:41
# @Author  : Yu-dong Wu
# @Site    : 
# @File    : common.py
# @Software: PyCharm
# @profile : some common utils
import pprint

def print_config(config):
    pp = pprint.PrettyPrinter(indent=1, width=80,)
    pp.pprint(config)