import tensorflow as tf
import numpy as np
from abc import ABC, abstractmethod, ABCMeta

class IOUHelper:

    def __init__(self):
        self.anchors_size=np.array([3.9, 1.6, 1.5])

    def get_iou(self, truth2, predictions2):
        truth = truth2[:, :, :, :, :8]
        predictions3 = predictions2[:, :, :, :, :8]

        # mins = np.array([-0.5, -0.5, 0, 0.8, 0.3, 0.13, -1.1, -1.1])
        # maxs = np.array([0.5, 0.5, 1, 2.6, 1.4, 0.82, 1.1, 1.1])
        mins = np.array([0, 0, 0, -0.1, -0.1, -0.1, -1.1, -1.1])
        maxs = np.array([0, 0, 0, 3, 2, 2, 1.1, 1.1])
        mins = np.expand_dims(mins, [0, 1, 2])
        maxs = np.expand_dims(maxs, [0, 1, 2])
        
        
        truth3 = ((truth + 1) / 2) * (maxs - mins) + mins 
        predictions4 = ((tf.nn.tanh(predictions3) + 1) / 2) * (maxs - mins) + mins 

        size_true = tf.exp(truth3[:, :, :, :, 3:6])
        size_pred = tf.exp(predictions4[:, :, :, :, 3:6])

        predictions = tf.math.sigmoid(predictions3)-0.5

        x = truth[:, :, :, :, 0]
        x_ = predictions[:, :, :, :, 0]
        y = truth[:, :, :, :, 1]
        y_ = predictions[:, :, :, :, 1]
        z = truth[:, :, :, :, 2]
        z_ = predictions[:, :, :, :, 2]


        x1 = x + size_true[:, :, :, :, 0]/2
        x2 = x - size_true[:, :, :, :, 0]/2
        y1 = y + size_true[:, :, :, :, 1]/2
        y2 = y - size_true[:, :, :, :, 1]/2
        z1 = z + size_true[:, :, :, :, 2]/2
        z2 = z - size_true[:, :, :, :, 2]/2

        x1_ = x_ + size_pred[:, :, :, :, 0]/2
        x2_ = x_ - size_pred[:, :, :, :, 0]/2
        y1_ = y_ + size_pred[:, :, :, :, 1]/2
        y2_ = y_ - size_pred[:, :, :, :, 1]/2
        z1_ = z_ + size_pred[:, :, :, :, 2]/2
        z2_ = z_ - size_pred[:, :, :, :, 2] / 2

        area_g = size_true[:, :, :, :, 0] * size_true[:, :, :, :, 1]
        area_d = size_pred[:, :, :, :, 0] * size_pred[:, :, :, :, 1]
        h_g = size_true[:, :, :, :, 2]
        h_d = size_pred[:, :, :, :, 2]

        area_overlap = tf.maximum(0., (tf.minimum(x1, x1_) - tf.maximum(x2, x2_))) * tf.maximum(0., (tf.minimum(y1, y1_) - tf.maximum(y2, y2_)))
        h_overlap = tf.maximum(0., tf.minimum(z1, z1_) - tf.maximum(z2, z2_))

        iou = (area_overlap * h_overlap) / (area_g*h_g + area_d*h_d - area_overlap*h_overlap + 1e-8)
        iou = tf.where(tf.greater_equal(iou, 0), iou, tf.zeros_like(truth2[:, :, :, :, 8]))
        iou = tf.where(tf.greater_equal(truth2[:, :, :, :, 8],0.5), iou, tf.zeros_like(truth2[:, :, :, :, 8]))
        iou = tf.reduce_sum(iou) / (tf.math.count_nonzero(truth2[:, :, :, :, 8], dtype=tf.float32)+1e-8)

        iou_2d = (area_overlap) / (area_g + area_d - area_overlap + 1e-8)
        iou_2d = tf.where(tf.greater_equal(iou_2d, 0), iou_2d, tf.zeros_like(truth2[:, :, :, :, 8]))
        iou_2d = tf.where(tf.greater_equal(truth2[:, :, :, :, 8],0.5), iou_2d, tf.zeros_like(truth2[:, :, :, :, 8]))
        iou_2d = tf.reduce_sum(iou_2d) / (tf.math.count_nonzero(truth2[:, :, :, :, 8], dtype=tf.float32)+1e-8)

        return iou, iou_2d


    def get_iou_loc(self, truth2, predictions2):
        truth = truth2[:, :, :, :, :8]
        predictions3 = predictions2[:, :, :, :, :8]

        # mins = np.array([-0.5, -0.5, 0, 0.8, 0.3, 0.13, -1.1, -1.1])
        # maxs = np.array([0.5, 0.5, 1, 2.6, 1.4, 0.82, 1.1, 1.1])
        mins = np.array([0, 0, 0, -0.1, -0.1, -0.1, -1.1, -1.1])
        maxs = np.array([0, 0, 0, 3, 2, 2, 1.1, 1.1])
        mins = np.expand_dims(mins, [0, 1, 2])
        maxs = np.expand_dims(maxs, [0, 1, 2])
        
        
        truth3 = ((truth + 1) / 2) * (maxs - mins) + mins 
        predictions4 = ((tf.nn.tanh(predictions3) + 1) / 2) * (maxs - mins) + mins 

        size_true = tf.exp(truth3[:, :, :, :, 3:6])
        size_pred = tf.exp(predictions4[:, :, :, :, 3:6])

        predictions = tf.math.sigmoid(predictions3)-0.5

        x = truth[:, :, :, :, 0]
        x_ = predictions[:, :, :, :, 0]
        y = truth[:, :, :, :, 1]
        y_ = predictions[:, :, :, :, 1]
        z = truth[:, :, :, :, 2]
        z_ = predictions[:, :, :, :, 2]

        x1 = x + size_true[:, :, :, :, 0]/2
        x2 = x - size_true[:, :, :, :, 0]/2
        y1 = y + size_true[:, :, :, :, 1]/2
        y2 = y - size_true[:, :, :, :, 1]/2
        z1 = z + size_true[:, :, :, :, 2]/2
        z2 = z - size_true[:, :, :, :, 2]/2

        x1_ = x_ + size_true[:, :, :, :, 0]/2
        x2_ = x_ - size_true[:, :, :, :, 0]/2
        y1_ = y_ + size_true[:, :, :, :, 1]/2
        y2_ = y_ - size_true[:, :, :, :, 1]/2
        z1_ = z_ + size_true[:, :, :, :, 2]/2
        z2_ = z_ - size_true[:, :, :, :, 2]/2

        area_g = size_true[:, :, :, :, 0] * size_true[:, :, :, :, 1]
        area_d = size_true[:, :, :, :, 0] * size_true[:, :, :, :, 1]
        h_g = size_true[:, :, :, :, 2]
        h_d = size_true[:, :, :, :, 2]

        area_overlap = tf.maximum(0., (tf.minimum(x1, x1_) - tf.maximum(x2, x2_))) * tf.maximum(0., (tf.minimum(y1, y1_) - tf.maximum(y2, y2_)))
        h_overlap = tf.maximum(0., tf.minimum(z1, z1_) - tf.maximum(z2, z2_))

        iou_loc = (area_overlap * h_overlap) / (area_g*h_g + area_d*h_d - area_overlap*h_overlap + 1e-8)
        iou_loc = tf.where(tf.greater_equal(iou_loc, 0), iou_loc, tf.zeros_like(truth2[:, :, :, :, 8]))
        iou_loc = tf.where(tf.greater_equal(truth2[:, :, :, :, 8],0.5), iou_loc, tf.zeros_like(truth2[:, :, :, :, 8]))
        iou_loc = tf.reduce_sum(iou_loc) / (tf.math.count_nonzero(truth2[:, :, :, :, 8], dtype=tf.float32)+1e-8)

        return iou_loc


    def get_iou_dim(self, truth2, predictions2):
        truth = truth2[:, :, :, :, :8]
        predictions3 = predictions2[:, :, :, :, :8]

        # mins = np.array([-0.5, -0.5, 0, 0.8, 0.3, 0.13, -1.1, -1.1])
        # maxs = np.array([0.5, 0.5, 1, 2.6, 1.4, 0.82, 1.1, 1.1])
        mins = np.array([0, 0, 0, -0.1, -0.1, -0.1, -1.1, -1.1])
        maxs = np.array([0, 0, 0, 3, 2, 2, 1.1, 1.1])
        mins = np.expand_dims(mins, [0, 1, 2])
        maxs = np.expand_dims(maxs, [0, 1, 2])
        
        
        truth3 = ((truth + 1) / 2) * (maxs - mins) + mins 
        predictions4 = ((tf.nn.tanh(predictions3) + 1) / 2) * (maxs - mins) + mins 

        size_true = tf.exp(truth3[:, :, :, :, 3:6])
        size_pred = tf.exp(predictions4[:, :, :, :, 3:6])

        predictions = tf.math.sigmoid(predictions3)-0.5

        x = truth[:, :, :, :, 0]
        x_ = predictions[:, :, :, :, 0]
        y = truth[:, :, :, :, 1]
        y_ = predictions[:, :, :, :, 1]
        z = truth[:, :, :, :, 2]
        z_ = predictions[:, :, :, :, 2]


        x1 = x + size_true[:, :, :, :, 0]/2
        x2 = x - size_true[:, :, :, :, 0]/2
        y1 = y + size_true[:, :, :, :, 1]/2
        y2 = y - size_true[:, :, :, :, 1]/2
        z1 = z + size_true[:, :, :, :, 2]/2
        z2 = z - size_true[:, :, :, :, 2]/2

        x1_ = x + size_pred[:, :, :, :, 0]/2
        x2_ = x - size_pred[:, :, :, :, 0]/2
        y1_ = y + size_pred[:, :, :, :, 1]/2
        y2_ = y - size_pred[:, :, :, :, 1]/2
        z1_ = z + size_pred[:, :, :, :, 2]/ 2
        z2_ = z - size_pred[:, :, :, :, 2]/2

        area_g = size_true[:, :, :, :, 0] * size_true[:, :, :, :, 1]
        area_d = size_pred[:, :, :, :, 0] * size_pred[:, :, :, :, 1]
        h_g = size_true[:, :, :, :, 2]
        h_d = size_pred[:, :, :, :, 2]

        area_overlap = tf.maximum(0., (tf.minimum(x1, x1_) - tf.maximum(x2, x2_))) * tf.maximum(0., (tf.minimum(y1, y1_) - tf.maximum(y2, y2_)))
        h_overlap = tf.maximum(0., tf.minimum(z1, z1_) - tf.maximum(z2, z2_))

        iou_dim = (area_overlap * h_overlap) / (area_g*h_g + area_d*h_d - area_overlap*h_overlap + 1e-8)
        iou_dim = tf.where(tf.greater_equal(iou_dim, 0), iou_dim, tf.zeros_like(truth2[:, :, :, :, 8]))
        iou_dim = tf.where(tf.greater_equal(truth2[:, :, :, :, 8],0.5), iou_dim, tf.zeros_like(truth2[:, :, :, :, 8]))
        iou_dim = tf.reduce_sum(iou_dim) / (tf.math.count_nonzero(truth2[:, :, :, :, 8], dtype=tf.float32)+1e-8)

        return iou_dim


    def get_iou_loc_x(self, truth2, predictions2):
        truth = truth2[:, :, :, :, :8]
        predictions = predictions2[:, :, :, :, :8]

        mins = np.array([-0.5, -0.5, 0, 0.7, 0.1, 0.1, -1.1, -1.1])
        maxs = np.array([0.5, 0.5, 1, 1.9, 0.75, 0.91, 1.1, 1.1])

        mins = np.expand_dims(mins, [0, 1, 2])
        maxs = np.expand_dims(maxs, [0, 1, 2])

        truth = (truth + 1) / 2
        truth = truth * (maxs - mins) + mins 
        predictions = (predictions + 1) / 2
        predictions = predictions * (maxs - mins) + mins 

        size_true = tf.exp(truth[:, :, :, :, 3:6])
        size_pred = tf.exp(predictions[:, :, :, :, 3:6])

        x = truth[:, :, :, :, 0]
        x_ = predictions[:, :, :, :, 0]
        y = truth[:, :, :, :, 1]
        y_ = y
        z = truth[:, :, :, :, 2] + 1. - 0.5
        z_ = z

        x1 = x + size_true[:, :, :, :, 0]/2
        x2 = x - size_true[:, :, :, :, 0]/2
        y1 = y + size_true[:, :, :, :, 1]/2
        y2 = y - size_true[:, :, :, :, 1]/2
        z1 = z
        z2 = z - size_true[:, :, :, :, 2]

        x1_ = x_ + size_true[:, :, :, :, 0]/2
        x2_ = x_ - size_true[:, :, :, :, 0]/2
        y1_ = y_ + size_true[:, :, :, :, 1]/2
        y2_ = y_ - size_true[:, :, :, :, 1]/2
        z1_ = z_
        z2_ = z_ - size_true[:, :, :, :, 2]

        area_g = size_true[:, :, :, :, 0] * size_true[:, :, :, :, 1]
        area_d = size_true[:, :, :, :, 0] * size_true[:, :, :, :, 1]
        h_g = size_true[:, :, :, :, 2]
        h_d = size_true[:, :, :, :, 2]

        area_overlap = tf.maximum(0., (tf.minimum(x1, x1_) - tf.maximum(x2, x2_))) * tf.maximum(0., (tf.minimum(y1, y1_) - tf.maximum(y2, y2_)))
        h_overlap = tf.maximum(0., tf.minimum(z1, z1_) - tf.maximum(z2, z2_))

        iou_loc_x = (area_overlap * h_overlap) / (area_g*h_g + area_d*h_d - area_overlap*h_overlap + 1e-8)
        iou_loc_x = tf.where(tf.greater_equal(iou_loc_x, 0), iou_loc_x, tf.zeros_like(truth2[:, :, :, :, 8]))
        iou_loc_x = tf.where(tf.greater_equal(truth2[:, :, :, :, 8],0.5), iou_loc_x, tf.zeros_like(truth2[:, :, :, :, 8]))
        iou_loc_x = tf.reduce_sum(iou_loc_x) / (tf.math.count_nonzero(truth2[:, :, :, :, 8], dtype=tf.float32)+1e-8)

        return iou_loc_x


    def get_iou_loc_y(self, truth2, predictions2):
        truth = truth2[:, :, :, :, :8]
        predictions = predictions2[:, :, :, :, :8]

        mins = np.array([-0.5, -0.5, 0, 0.7, 0.1, 0.1, -1.1, -1.1])
        maxs = np.array([0.5, 0.5, 1, 1.9, 0.75, 0.85, 1.1, 1.1])

        mins = np.expand_dims(mins, [0, 1, 2])
        maxs = np.expand_dims(maxs, [0, 1, 2])

        
        truth = (truth + 1) / 2
        truth = truth * (maxs - mins) + mins 
        predictions = (predictions + 1) / 2
        predictions = predictions * (maxs - mins) + mins 

        size_true = tf.exp(truth[:, :, :, :, 3:6])
        size_pred = tf.exp(predictions[:, :, :, :, 3:6])

        x = truth[:, :, :, :, 0]
        x_ = x
        y = truth[:, :, :, :, 1]
        y_ = predictions[:, :, :, :, 1]
        z = truth[:, :, :, :, 2] + 1. - 0.5
        z_ = z

        x1 = x + size_true[:, :, :, :, 0]/2
        x2 = x - size_true[:, :, :, :, 0]/2
        y1 = y + size_true[:, :, :, :, 1]/2
        y2 = y - size_true[:, :, :, :, 1]/2
        z1 = z
        z2 = z - size_true[:, :, :, :, 2]

        x1_ = x_ + size_true[:, :, :, :, 0]/2
        x2_ = x_ - size_true[:, :, :, :, 0]/2
        y1_ = y_ + size_true[:, :, :, :, 1]/2
        y2_ = y_ - size_true[:, :, :, :, 1]/2
        z1_ = z_
        z2_ = z_ - size_true[:, :, :, :, 2]

        area_g = size_true[:, :, :, :, 0] * size_true[:, :, :, :, 1]
        area_d = size_true[:, :, :, :, 0] * size_true[:, :, :, :, 1]
        h_g = size_true[:, :, :, :, 2]
        h_d = size_true[:, :, :, :, 2]

        area_overlap = tf.maximum(0., (tf.minimum(x1, x1_) - tf.maximum(x2, x2_))) * tf.maximum(0., (tf.minimum(y1, y1_) - tf.maximum(y2, y2_)))
        h_overlap = tf.maximum(0., tf.minimum(z1, z1_) - tf.maximum(z2, z2_))

        iou_loc_y = (area_overlap * h_overlap) / (area_g*h_g + area_d*h_d - area_overlap*h_overlap + 1e-8)
        iou_loc_y = tf.where(tf.greater_equal(iou_loc_y, 0), iou_loc_y, tf.zeros_like(truth2[:, :, :, :, 8]))
        iou_loc_y = tf.where(tf.greater_equal(truth2[:, :, :, :, 8],0.5), iou_loc_y, tf.zeros_like(truth2[:, :, :, :, 8]))
        iou_loc_y = tf.reduce_sum(iou_loc_y) / (tf.math.count_nonzero(truth2[:, :, :, :, 8], dtype=tf.float32)+1e-8)

        return iou_loc_y

    def get_iou_loc_z(self, truth2, predictions2):
        truth = truth2[:, :, :, :, :8]
        predictions = predictions2[:, :, :, :, :8]

        mins = np.array([-0.5, -0.5, 0, 0.7, 0.1, 0.1, -1.1, -1.1])
        maxs = np.array([0.5, 0.5, 1, 1.9, 0.75, 0.85, 1.1, 1.1])

        mins = np.expand_dims(mins, [0, 1, 2])
        maxs = np.expand_dims(maxs, [0, 1, 2])
        
        truth = (truth + 1) / 2
        truth = truth * (maxs - mins) + mins 
        predictions = (predictions + 1) / 2
        predictions = predictions * (maxs - mins) + mins 

        size_true = tf.exp(truth[:, :, :, :, 3:6])
        size_pred = tf.exp(predictions[:, :, :, :, 3:6])

        x = truth[:, :, :, :, 0]
        x_ = x
        y = truth[:, :, :, :, 1]
        y_ = y
        z = truth[:, :, :, :, 2] + 1. - 0.5
        z_ = predictions[:, :, :, :, 2]+ 1. - 0.5

        x1 = x + size_true[:, :, :, :, 0]/2
        x2 = x - size_true[:, :, :, :, 0]/2
        y1 = y + size_true[:, :, :, :, 1]/2
        y2 = y - size_true[:, :, :, :, 1]/2
        z1 = z
        z2 = z - size_true[:, :, :, :, 2]

        x1_ = x_ + size_true[:, :, :, :, 0]/2
        x2_ = x_ - size_true[:, :, :, :, 0]/2
        y1_ = y_ + size_true[:, :, :, :, 1]/2
        y2_ = y_ - size_true[:, :, :, :, 1]/2
        z1_ = z_
        z2_ = z_ - size_true[:, :, :, :, 2]

        area_g = size_true[:, :, :, :, 0] * size_true[:, :, :, :, 1]
        area_d = size_true[:, :, :, :, 0] * size_true[:, :, :, :, 1]
        h_g = size_true[:, :, :, :, 2]
        h_d = size_true[:, :, :, :, 2]

        area_overlap = tf.maximum(0., (tf.minimum(x1, x1_) - tf.maximum(x2, x2_))) * tf.maximum(0., (tf.minimum(y1, y1_) - tf.maximum(y2, y2_)))
        h_overlap = tf.maximum(0., tf.minimum(z1, z1_) - tf.maximum(z2, z2_))

        iou_loc_z = (area_overlap * h_overlap) / (area_g*h_g + area_d*h_d - area_overlap*h_overlap + 1e-8)
        iou_loc_z = tf.where(tf.greater_equal(iou_loc_z, 0), iou_loc_z, tf.zeros_like(truth2[:, :, :, :, 8]))
        iou_loc_z = tf.where(tf.greater_equal(truth2[:, :, :, :, 8],0.5), iou_loc_z, tf.zeros_like(truth2[:, :, :, :, 8]))
        iou_loc_z = tf.reduce_sum(iou_loc_z) / (tf.math.count_nonzero(truth2[:, :, :, :, 8], dtype=tf.float32)+1e-8)

        return iou_loc_z