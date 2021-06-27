import tensorflow as tf
import argparse
import numpy as np
from tensorflow.python import debug as tf_debug

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
from training.Trainer import *
from training.segmentation_trainer import *
from training.detection_trainer import *
from training.detection_trainer_lr_find import *




class ModelTrainer(object):

    def __init__(self, model, data_base_path):
        self.model = model
        self.data_base_path = data_base_path

        self.fusion_trainer = FusionDetectionTrainer(self.model, self.data_base_path, None)
        self.bev_trainer = BEVDetectionTrainer(self.model, self.data_base_path, None)


    def train_bev(self, **kwargs):
        self.bev_trainer.train(restore=kwargs['restore'], 
                    epochs=kwargs['epochs'], 
                    num_samples=kwargs['num_samples'], 
                    training_per=kwargs['training_per'], 
                    random_seed=kwargs['random_seed'], 
                    training=kwargs['training'], 
                    batch_size=kwargs['batch_size'], 
                    save_steps=kwargs['save_steps'],
                    start_epoch=kwargs['start_epoch'],
                    augment=kwargs['augment'],
                    fusion=kwargs['train_fusion'])


    def train_fusion(self, **kwargs):
        self.fusion_trainer.train(restore=kwargs['restore'], 
                    epochs=kwargs['epochs'], 
                    num_samples=kwargs['num_samples'], 
                    training_per=kwargs['training_per'], 
                    random_seed=kwargs['random_seed'], 
                    training=kwargs['training'], 
                    batch_size=kwargs['batch_size'], 
                    save_steps=kwargs['save_steps'],
                    start_epoch=kwargs['start_epoch'],
                    augment=kwargs['augment'],
                    fusion=kwargs['train_fusion'])

   
