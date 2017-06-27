# -*- coding: utf-8 -*-
from tools_config import *
import os 
import tensorflow as tf

os.environ['GLOG_minloglevel'] = '3'  # 0 - debug 1 - info (still a LOT of outputs) 2 - warnings 3 - errors

import numpy as np

# rng = np.random.RandomState(481542)
# tf.set_random_seed(4815162342)
tf.set_random_seed(None)
rng = np.random.RandomState(None)

def gather_along_axis(data, indices, axis=0):
  if not axis:
    return tf.gather(data, indices)
  rank = data.shape.ndims
  perm = [axis] + list(range(1, axis)) + [0] + list(range(axis + 1, rank))
  
  return tf.transpose(tf.gather(tf.transpose(data, perm), indices), perm)

class AttributeDict(dict): 
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__