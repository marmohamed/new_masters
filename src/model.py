import tensorflow as tf
import argparse
import numpy as np
from tensorflow.python import debug as tf_debug
import os

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
from PCGrad_tf import *

class Model(object):

    def __init__(self, graph=None, **params):
        self.CONST = Const()
        self.graph = graph
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
            'train_cls': True,
            'train_reg': True,
            'fusion': True,
            'mse_loss': False,
            'res_blocks': 0,
            'res_blocks_image': 1,
            'train_loc': 1,
            'train_dim': 1,
            'train_theta': 1,
            'train_dir': 1
        }
        for k in params:
            if k in defaults:
                defaults[k] = params[k]
        return defaults


    def __build_model(self):

        self.debug_layers = {}

        if self.graph is None:
            self.graph = tf.Graph()

        with self.graph.as_default():
                self.is_training = tf.keras.laters.Input(shape=(), dtype=tf.bool, name='is_training')

                img_size_1 = 370
                img_size_2 = 1224
                c_dim = 3
                self.train_inputs_rgb = tf.keras.laters.Input(
                                                    dtype=tf.float32,
                                                    shape=[None, img_size_1, img_size_2, c_dim], 
                                                    name='train_inputs_rgb')

                img_size_1 = 448
                img_size_2 = 512
                c_dim = 36
                self.train_inputs_lidar = tf.keras.laters.Input(
                                    dtype=tf.float32,
                                    shape=[None, img_size_1, img_size_2, c_dim], 
                                    name='train_inputs_lidar')

                self.y_true = tf.keras.laters.Input(dtype=tf.float32, shape=(None, 112, 128, 2, 13)) # target

                self.y_true_img = tf.keras.laters.Input(dtype=tf.float32, shape=(None, 24, 78, 2)) # target
                self.train_fusion_rgb = tf.keras.laters.Input(dtype=tf.bool, shape=())

                
                with tf.variable_scope("image_branch"):
                    self.cnn = ResNetBuilder().build(branch=self.CONST.IMAGE_BRANCH, img_height=370, img_width=1224, img_channels=3)
                    self.cnn.build_model(self.train_inputs_rgb, is_training=self.is_training)
                    
                with tf.variable_scope("lidar_branch"):
                    self.cnn_lidar = ResNetBuilder().build(branch=self.CONST.BEV_BRANCH, img_height=512, img_width=448, img_channels=40)
                    self.cnn_lidar.build_model(self.train_inputs_lidar, is_training=self.is_training)
                
            
                self.cnn_lidar.res_groups2 = self.cnn_lidar.res_groups
                self.cnn.res_groups2 = self.cnn.res_groups


                fpn_lidar = FPN(self.cnn_lidar.res_groups2, scope="fpn_lidar", is_training=self.is_training)
                fpn_lidar[0] = maxpool2d(fpn_lidar[0], scope="fpn_lidar_maxpool_0")
                        # fpn_lidar[-1] = upsample(fpn_lidar[-1], scope="fpn_lidar_upsample_0", filters=128, use_deconv=True, kernel_size=4)

                fpn_lidar = tf.concat(fpn_lidar, axis=-1)

                fpn_lidar1 = fpn_lidar[:]
                        # fpn_lidar2 = fpn_lidar[:]

                num_conv_blocks=4
                for i in range(0, num_conv_blocks):
                    fpn_lidar1 = conv(fpn_lidar1, 96, kernel=3, stride=1, padding='SAME', use_bias=True, scope='conv_post_fpn_11_'+str(i))
                    fpn_lidar1 = batch_norm(fpn_lidar1, is_training=self.is_training, scope='bn_post_fpn_11_' + str(i))
                    fpn_lidar1 = relu(fpn_lidar1)
                    self.debug_layers['fpn_lidar1_output_post_conv_1_'+str(i)] = fpn_lidar1

                        # num_conv_blocks=2
                        # for i in range(0, num_conv_blocks):
                        #     fpn_lidar2 = conv(fpn_lidar2, 128, kernel=3, stride=1, padding='SAME', use_bias=True, scope='conv_post_fpn_12_'+str(i))
                        #     fpn_lidar2 = batch_norm(fpn_lidar2, is_training=self.is_training, scope='bn_post_fpn_12_' + str(i))
                        #     fpn_lidar2 = relu(fpn_lidar2)
                        #     self.debug_layers['fpn_lidar2_output_post_conv_1_'+str(i)] = fpn_lidar2
                     

                if self.params['focal_loss']:
                    final_output_1_6 = conv(fpn_lidar1, 7, kernel=1, stride=1, padding='SAME', use_bias=True, scope='conv_out_1', use_ws_reg=False)
                    inal_output_1_7 = conv(fpn_lidar1, 1, kernel=1, stride=1, padding='SAME', use_bias=True, scope='conv_out_1_7', use_ws_reg=False)
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


                self.loc_weight = tf.get_variable('loc_weight', shape=(), initializer=tf.constant_initializer(1), dtype=tf.float32)
                self.dim_weight = tf.get_variable('dim_weight', shape=(), initializer=tf.constant_initializer(1), dtype=tf.float32)
                self.theta_weight = tf.get_variable('theta_weight', shape=(), initializer=tf.constant_initializer(1), dtype=tf.float32)
                self.cls_weight = tf.get_variable('cls_weight', shape=(), initializer=tf.constant_initializer(1), dtype=tf.float32)
                        

                with tf.variable_scope('Loss'):
                cls_loss_instance = ClsLoss('classification_loss')
                reg_loss_instance = RegLoss('regression_loss')
                loss_calculator = LossCalculator()
                loss_params = {'focal_loss': self.params['focal_loss'], 'weight': self.params['weight_loss'], 'mse': self.params['mse_loss']}
                self.classification_loss, self.loc_reg_loss, self.dim_reg_loss,\
                                    self.theta_reg_loss, self.dir_reg_loss, self.corners_loss, self.oclussion_loss,\
                                    self.precision, self.recall, self.iou, self.iou_2d, self.iou_loc, self.iou_dim, self.theta_accuracy = loss_calculator(
                                                            self.y_true,
                                                            self.final_output, 
                                                            cls_loss_instance, 
                                                            reg_loss_instance,
                                                            **loss_params)

                self.weight_cls = tf.placeholder(tf.float32, shape=())
                self.weight_dim = tf.placeholder(tf.float32, shape=())
                self.weight_loc = tf.placeholder(tf.float32, shape=())
                self.weight_theta = tf.placeholder(tf.float32, shape=())
                self.weight_dir = tf.placeholder(tf.float32, shape=())


                self.regression_loss_bev = 0
                if self.params['train_loc'] == 1:
                            self.regression_loss_bev += 10 * self.weight_loc * self.loc_reg_loss 
                if self.params['train_dim'] == 1:
                            self.regression_loss_bev += 10 * self.weight_dim * self.dim_reg_loss 
                if self.params['train_theta'] == 1:
                            self.regression_loss_bev += 5 * self.weight_theta * self.theta_reg_loss 

                        
                self.model_loss_bev = 0
                if self.params['train_cls']:
                            self.model_loss_bev +=  1 * self.weight_cls * self.classification_loss
                if self.params['train_reg']:
                            self.model_loss_bev +=  1 * self.regression_loss_bev

                self.model_loss_bev += 5 * (self.weight_loc + self.weight_dim)  * self.corners_loss
                self.model_loss_bev += self.weight_dir * self.dir_reg_loss
                self.model_loss_bev += 0.5 * self.oclussion_loss

                self.regression_loss = self.regression_loss_bev
                self.model_loss = self.model_loss_bev

                self.losses = [ 1 * self.weight_loc * self.loc_reg_loss , 
                                        3 * self.weight_dim * self.dim_reg_loss, 
                                        2 * self.weight_theta * self.theta_reg_loss, 
                                        1 * self.weight_cls * self.classification_loss]

               
               
                     
                self.global_step = tf.Variable(0, name='global_step', trainable=False)

                
                self.decay_rate = tf.train.exponential_decay(self.params['lr'], self.global_step, self.params['decay_steps'], 
                                                            self.params['decay_rate'], self.params['staircase'])  

                self.learning_rate_placeholder = tf.placeholder(tf.float32, [], name='learning_rate')
                self.opt_lidar = tf.train.AdamOptimizer(self.learning_rate_placeholder)
                self.train_op_lidar = self.opt_lidar.minimize(self.model_loss,\
                                                                            var_list=self.lidar_only_vars,\
                                                                            global_step=self.global_step)



                self.saver = tf.train.Saver(max_to_keep=1)

                self.best_saver = tf.train.Saver(max_to_keep=1)

                self.lr_summary = tf.summary.scalar('learning_rate', tf.squeeze(self.decay_rate))
                self.model_loss_batches_summary = tf.summary.scalar('model_loss_batches', self.model_loss)
                self.cls_loss_batches_summary = tf.summary.scalar('classification_loss_batches', self.classification_loss)
                self.reg_loss_batches_summary = tf.summary.scalar('regression_loss_batches', self.regression_loss)
                self.loc_reg_loss_batches_summary = tf.summary.scalar('loc_regression_loss_batches', self.loc_reg_loss)
                self.dim_reg_loss_batches_summary = tf.summary.scalar('dim_regression_loss_batches', self.dim_reg_loss)
                self.theta_reg_loss_batches_summary = tf.summary.scalar('theta_regression_loss_batches', self.theta_reg_loss)
                self.dir_reg_loss_batches_summary = tf.summary.scalar('dir_regression_loss_batches', self.dir_reg_loss)
                self.corners_loss_batches_summary = tf.summary.scalar('corners_regression_loss_batches', self.corners_loss)
                self.occlusion_loss_batches_summary = tf.summary.scalar('occlusion_regression_loss_batches', self.oclussion_loss)

                self.precision_summary = tf.summary.scalar('precision_batches', self.precision)
                self.recall_summary = tf.summary.scalar('recall_batches', self.recall)

                self.iou_summary = tf.summary.scalar('iou_batches', self.iou)
                self.iou_2d_summary = tf.summary.scalar('iou_2d_batches', self.iou_2d)
                self.iou_loc_summary = tf.summary.scalar('iou_loc_batches', self.iou_loc)
                self.iou_dim_summary = tf.summary.scalar('iou_dim_batches', self.iou_dim)
                self.theta_accuracy_summary = tf.summary.scalar('theta_accuracy_batches', self.theta_accuracy)

                self.cls_weight_summary = tf.summary.scalar('cls_weight_summary', self.weight_cls)
                self.loc_weight_summary = tf.summary.scalar('loc_weight_summary', self.weight_loc)
                self.dim_weight_summary = tf.summary.scalar('dim_weight_summary', self.weight_dim)
                self.theta_weight_summary = tf.summary.scalar('theta_weight_summary', self.weight_theta)


               

              
                self.merged = tf.summary.merge([self.lr_summary, self.model_loss_batches_summary, \
                                            self.cls_loss_batches_summary, self.reg_loss_batches_summary,\
                                            self.loc_reg_loss_batches_summary, self.dim_reg_loss_batches_summary,\
                                            self.theta_reg_loss_batches_summary,\
                                            self.precision_summary, self.recall_summary,\
                                            self.iou_summary, self.iou_2d_summary, self.iou_loc_summary, self.iou_dim_summary,\
                                            self.theta_accuracy_summary,\
                                            self.cls_weight_summary, self.loc_weight_summary, self.dim_weight_summary,self.theta_weight_summary,\
                                            self.dir_reg_loss_batches_summary, self.corners_loss_batches_summary,\
                                            self.occlusion_loss_batches_summary
                                            ])



