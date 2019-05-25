import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
import bi_lstm_crf
import pickle
import os

ops.reset_default_graph()
data_folder_name = 'temp'
data_path_name = 'data'