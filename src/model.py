import tensorflow as tf
import argparse
import numpy as np
import os

from utils.constants import *
from utils.utils import *
from utils.anchors import *
from utils.nms import *

from loss.losses_v2 import *

from models.ResNetBuilder import *
from models.ResnetImage import *
from models.ResnetLidarBEV import *
from models.ResnetLidarFV import *

from FPN.FPN import *

from Fusion.FusionLayer import *
from PCGrad_tf import *

class Model(object):

    def __init__(self, graph=None, **params):
        self.CONST = Const()
        self.params = self.__prepare_parameters(**params)
        self.__build_model()


    def __prepare_parameters(self, **params):
        defaults = {
            'focal_loss': True,
            'weight_loss': False,
            'focal_init': -1.99,
            'lr': 5e-4, 
            'decay_steps': 5000,
            'decay_rate': 0.9,
            'staircase': False,
            'mse_loss': False,
            'res_blocks': 0,
            'res_blocks_image': 1,
        }
        for k in params:
            if k in defaults:
                defaults[k] = params[k]
        return defaults


    def __build_model(self):


                img_size_1 = 448
                img_size_2 = 512
                c_dim = 36
                self.train_inputs_lidar = tf.keras.layers.Input(
                                    dtype=tf.float32,
                                    shape=[img_size_1, img_size_2, c_dim], 
                                    name='train_inputs_lidar')

                self.y_true = tf.keras.layers.Input(dtype=tf.float32, shape=(112, 128, 2, 13)) # target


                self.cnn_lidar = ResNetBuilder().build(branch=self.CONST.BEV_BRANCH, img_height=512, img_width=448, img_channels=40)
                self.cnn_lidar.build_model(self.train_inputs_lidar)
                
            
                self.cnn_lidar.res_groups2 = self.cnn_lidar.res_groups


                # fpn_lidar = FPN(self.cnn_lidar.res_groups2, scope="fpn_lidar")
                # fpn_lidar[0] = maxpool2d(fpn_lidar[0], scope="fpn_lidar_maxpool_0")
                #         # fpn_lidar[-1] = upsample(fpn_lidar[-1], scope="fpn_lidar_upsample_0", filters=128, use_deconv=True, kernel_size=4)

                # fpn_lidar = tf.concat(fpn_lidar, axis=-1)

                fpn_lidar1 = self.cnn_lidar.res_groups2[-1]

                num_conv_blocks=4
                for i in range(0, num_conv_blocks):
                    fpn_lidar1 = conv(fpn_lidar1, 96, kernel=3, stride=1, padding='SAME', use_bias=True, scope='conv_post_fpn_11_'+str(i))
                    fpn_lidar1 = batch_norm(fpn_lidar1, scope='bn_post_fpn_11_' + str(i))
                    fpn_lidar1 = relu(fpn_lidar1)
                    self.debug_layers['fpn_lidar1_output_post_conv_1_'+str(i)] = fpn_lidar1

             
                if self.params['focal_loss']:
                    final_output_1_6 = conv(fpn_lidar1, 7, kernel=1, stride=1, padding='SAME', use_bias=True, scope='conv_out_1', use_ws_reg=False)
                    final_output_1_7 = conv(fpn_lidar1, 1, kernel=1, stride=1, padding='SAME', use_bias=True, scope='conv_out_1_7', use_ws_reg=False)
                    final_output_1_8 = conv(fpn_lidar1, 1, kernel=1, stride=1, padding='SAME', use_bias=True, scope='conv_out_1_8', focal_init=self.params['focal_init'], use_ws_reg=False)
                    final_output_1_13 = conv(fpn_lidar1, 4, kernel=1, stride=1, padding='SAME', use_bias=True, scope='conv_out_1_13', use_ws_reg=False)

                    final_output_2_6 = conv(fpn_lidar1, 7, kernel=1, stride=1, padding='SAME', use_bias=True, scope='conv_out_2', use_ws_reg=False)
                    final_output_2_7 = conv(fpn_lidar1, 1, kernel=1, stride=1, padding='SAME', use_bias=True, scope='conv_out_2_7', use_ws_reg=False)
                    final_output_2_8 = conv(fpn_lidar1, 1, kernel=1, stride=1, padding='SAME', use_bias=True, scope='conv_out_2_8', focal_init=self.params['focal_init'], use_ws_reg=False)
                    final_output_2_13 = conv(fpn_lidar1, 4, kernel=1, stride=1, padding='SAME', use_bias=True, scope='conv_out_2_13', use_ws_reg=False)

                            
                    final_output_1 = tf.concat([final_output_1_6, final_output_1_7, final_output_1_8, final_output_1_13], -1)
                    final_output_2 = tf.concat([final_output_2_6, final_output_2_7, final_output_2_8, final_output_2_13], -1)

                    final_output_1 = tf.expand_dims(final_output_1, 3)
                    final_output_2 = tf.expand_dims(final_output_2, 3)

                    self.final_output = tf.concat([final_output_1, final_output_2], 3)
                        
                else:
                    final_output_1 = conv(fpn_lidar, 9, kernel=1, stride=1, padding='SAME', use_bias=True, scope='conv_out_1')
                    final_output_2 = conv(fpn_lidar, 9, kernel=1, stride=1, padding='SAME', use_bias=True, scope='conv_out_2')

                self.debug_layers['final_layer'] = self.final_output

                def get_loss(truth, predictions):
                    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(truth[:, :, :, :, 8], predictions[:, :, :, :, 8])

                    reg_loss = tf.keras.losses.MeanSquaredError()
                    loss_fn = lambda t, p: tf.where(tf.greater_equal(truth[:, :, :, :, 8],0.5), reg_loss(t, p), tf.zeros_like(p))
                    # loc_ratios = [5, 5, 1]
                    reg_losses1 = [loss_fn(truth[:, :, :, :, i], tf.math.sigmoid(predictions[:, :, :, :, i])-0.5) for i in range(3)] 
                    reg_losses2 = [loss_fn(truth[:, :, :, :, i], tf.nn.tanh(predictions[:, :, :, :, i])) for i in range(3, 6)] 
                    # reg_losses3 = [loss_fn(truth[:, :, :, :, i] , predictions[:, :, :, :, i]) for i in range(6, 8)]
                    reg_losses3 = [loss_fn((truth[:, :, :, :, i] + np.pi/4) / (np.pi/2), tf.math.sigmoid(predictions[:, :, :, :, i])) for i in range(6, 7)]
                    loss_fn = lambda t, p: tf.where(tf.greater_equal(truth[:, :, :, :, 8],0.5), tf.nn.sigmoid_cross_entropy_with_logits(labels=t, logits=p), tf.zeros_like(p))
                    reg_losses4 = [loss_fn(truth[:, :, :, :, i], predictions[:, :, :, :, i]) for i in range(7, 8)]

                    c = (tf.math.count_nonzero(truth[:, :, :, :, 8], dtype=tf.float32)+1e-8)

                    loc_reg_loss = tf.reduce_sum(reg_losses1)  / c
                    dim_reg_loss = tf.reduce_sum(reg_losses2) / c
                    theta_reg_loss = tf.reduce_sum(reg_losses3) / c
                    dir_reg_loss = tf.reduce_sum(reg_losses4) / c

                    loss += loc_reg_loss + dim_reg_loss + theta_reg_loss + dir_reg_loss
                    return loss

                self.model_loss = get_loss

                     
                self.opt_lidar = tf.keras.optimizers.Adam(1e-3)

                self.model = tf.keras.models.Model(inputs=[self.train_inputs_lidar], outputs=[self.final_output])
                self.model.compile(optimizer=self.opt_lidar, loss=self.model_loss)

               