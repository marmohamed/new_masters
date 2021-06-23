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
from data.segmentation_dataset_loader import *
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


    def __prepare_dataset_feed_dict(self, dataset, train_fusion_rgb, is_training=True, batch_size=1):

        data = dataset.get_next(batch_size=batch_size)
       
        camera_tensor, lidar_tensor, label_tensor = data
        # d = {self.model.train_inputs_rgb: camera_tensor,
        #         self.model.train_inputs_lidar: lidar_tensor,
        #         self.model.y_true: label_tensor,                   
        #         self.model.train_fusion_rgb: train_fusion_rgb,
        #         self.model.is_training: is_training
        #         }
        d = {
                self.model.train_inputs_lidar: lidar_tensor,
                self.model.y_true: label_tensor,                   
                self.model.train_fusion_rgb: train_fusion_rgb,
                self.model.is_training: is_training
                }
        return d



    def train(self, restore=True, 
                    epochs=200, 
                    num_samples=None, 
                    training_per=0.5, 
                    random_seed=42, 
                    training=True, 
                    batch_size=1, 
                    save_steps=100,
                    start_epoch=0,
                    augment=True,
                    **kwargs):

        if self.dataset is None:
            self.dataset = DetectionDatasetLoader(self.data_base_path, num_samples, training_per, random_seed, training, True)

        self.eval_dataset = DetectionDatasetLoader(self.data_base_path, num_samples, training_per, random_seed, False, False)


       
        with self.model.graph.as_default():
                
                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True

                np.random.seed(random_seed)
                tf.set_random_seed(random_seed)

                with tf.Session(config=config) as sess:
                    if restore:
                        self.model.saver.restore(sess, tf.train.latest_checkpoint('./training_files/tmp/'))
                    else:
                        sess.run(tf.global_variables_initializer())

                    counter = 0

                    self.model.fit(self.dataset, epochs=epochs)


   
        

      

class BEVDetectionTrainer(DetectionTrainer):

    def _set_params(self):
        pass

        
class FusionDetectionTrainer(DetectionTrainer):
    
    def _set_params(self):
        pass


class EndToEndDetectionTrainer(DetectionTrainer):
    
    def _set_params(self):
        pass



