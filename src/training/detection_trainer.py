import tensorflow as tf
import argparse

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
from Fusion.FusionLayer import *
from data.detection_dataset_loader import *
from training.Trainer import *
from training import clr

from evaluation.evaluate import *
from utils.summary_images_utils import *

import math
import numpy.matlib as npm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io
import os



class DetectionTrainer(Trainer):

    __metaclass__ = ABCMeta

    def __init__(self, model, data_base_path, dataset):
        super(DetectionTrainer, self).__init__(model, data_base_path, dataset)
        self._set_params()
        self.count_not_best = 0
        self.base_lr = 0.0001
        self.count_not_best_cls = 0
        self.count_not_best_loc = 0
        self.count_not_best_dir = 0
        self.count_not_best_dim = 0
        self.count_not_best_theta = 0
        

    @abstractmethod
    def _set_params(self):
        pass



    def train(self, restore=None, 
                    ckpt_path=None,
                    epochs=1, 
                    random_seed=0, 
                    batch_size=4, 
                    start_epoch=0,
                    fusion=False):

        self.dataset = DetectionDatasetLoader(self.data_base_path, batch_size=batch_size, fusion=fusion, training=True)

        self.eval_dataset = DetectionDatasetLoader(self.data_base_path, batch_size=2, fusion=fusion, training=False)

        if restore is not None:
            self.model.model = tf.keras.models.load_model(ckpt_path)

        save_ckpt = tf.keras.callbacks.ModelCheckpoint(
                        ckpt_path,
                        monitor="val_loss",
                        verbose=0,
                        save_best_only=False,
                        save_weights_only=False,
                        mode="auto",
                        save_freq="epoch",
                        options=None,
                        **kwargs
                    )
        callbacks = [save_ckpt]
        self.model.model.fit(self.dataset, epochs=epochs, steps_per_epoch=7481//batch_size,
                             validation_data=self.eval_dataset, validation_steps=7481//2,
                             callbacks=callbacks)


   
        

      

class BEVDetectionTrainer(DetectionTrainer):

    def _set_params(self):
        pass

        
class FusionDetectionTrainer(DetectionTrainer):
    
    def _set_params(self):
        pass


class EndToEndDetectionTrainer(DetectionTrainer):
    
    def _set_params(self):
        pass



