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


    def train_bev(self, args):
        self.bev_trainer.train(restore=args.restore, 
                    ckpt_path=args.ckpt_path,
                    epochs=args.epochs, 
                    random_seed=args.random_seed, 
                    batch_size=args.batch_size, 
                    start_epoch=args.start_epoch,
                    fusion=args.train_fusion)


    def train_fusion(self, **kwargs):
        self.fusion_trainer.train(restore=args.restore, 
                    ckpt_path=args.ckpt_path,
                    epochs=args.epochs, 
                    random_seed=args.random_seed, 
                    batch_size=args.batch_size, 
                    start_epoch=args.start_epoch,
                    fusion=args.train_fusion)

   
