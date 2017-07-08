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