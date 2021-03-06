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
from data.detection_dataset_loader import *
from training.Trainer import *

from evaluation.evaluate import *
from utils.summary_images_utils import *

import math
import numpy.matlib as npm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io
import os



class DetectionTrainer(Trainer):

    def __init__(self, model, data_base_path, dataset):
        super(DetectionTrainer, self).__init__(model, data_base_path, dataset)
    


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
                        'model.{epoch:02d}-{val_loss:.4f}.h5',
                        monitor="val_loss",
                        verbose=1,
                        save_best_only=True,
                        mode="min",
                    )
        callbacks = [save_ckpt]
        self.model.model.fit(self.dataset, epochs=epochs, steps_per_epoch=3769//batch_size,
                             validation_data=self.eval_dataset, validation_steps=3712//2,
                             callbacks=callbacks)
        # self.model.model.fit(self.dataset, epochs=epochs, steps_per_epoch=1,
        #                      validation_data=self.eval_dataset, validation_steps=1,
        #                      callbacks=callbacks)



   
    


