import tensorflow as tf
import argparse
import numpy as np
from tensorflow.python import debug as tf_debug
from abc import ABC, abstractmethod, ABCMeta

from utils.constants import *
from utils.utils import *
from utils.anchors import *
from utils.nms import *
from loss.losses import *
from models.ResNetBuilder import *
from models.ResnetImage import *
from models.ResnetLidarBEV import *
from models.ResnetLidarFV import *
from FPN.FPN import *
from data.detection_dataset_loader import *


class Trainer(object):

    __metaclass__ = ABCMeta

    def __init__(self, model, data_base_path, dataset):
        self.model = model
        self.data_base_path = data_base_path
        self.dataset = dataset

    @abstractmethod
    def train(self, restore=None, 
                    ckpt_path=None,
                    epochs=1, 
                    random_seed=0, 
                    batch_size=4, 
                    start_epoch=0,
                    fusion=False):
        pass


    @abstractmethod
    def eval(self, sess, epoch, batch_size=1, **kwargs):
        pass


    