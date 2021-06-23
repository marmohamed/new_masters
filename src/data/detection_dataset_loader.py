import numpy as np
import cv2
from data.data_utils.velodyne_points import *
from utils.utils import *
import math
import os
import random
import glob

from data.data_utils.data_reader import *
from data.data_utils.target_utils import *
from data.data_utils.fv_utils import *
from data.dataset_loader import *

from scipy.spatial.transform import Rotation as R


class DetectionDatasetLoader(tf.keras.utils.Sequence):

    def __init__(self, batch_size=2, training=True, **kwargs):
        self.defaults = {
            'image_size': (370, 1224),
            'lidar_size': (448, 512, 40), 
            'anchors': np.array([3.9, 1.6, 1.5])
        }
        for k in kwargs:
            if k in defaults:
                self.defaults[k] = kwargs[k]

        self.batch_size = batch_size
        self.training = training
        self.augment = self.training

        self.list_camera_paths = glob.glob(os.path.join(self.base_path, 'image_2/*'))
        self.list_lidar_paths  = glob.glob(os.path.join(self.base_path, 'velodyne/*'))
        self.list_label_paths  = glob.glob(os.path.join(self.base_path, 'label_2/*'))
        self.list_calib_paths  = glob.glob(os.path.join(self.base_path, 'calib/*'))

        self.indexes = np.arange(len(self.list_lidar_paths))
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_lidar_paths) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        X, y = self.__data_generation(indexes)

        return X, y


    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_lidar_paths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def __data_generation(self, indexes):

        lidar_images = np.zeros((len(indexes), self.defaults['lidar_size'][0], self.defaults['lidar_size'][1], self.defaults['lidar_size'][1]))
        labels = np.zeros((len(indexes), 112, 128, 2, 13))

        for i in range(len(self.indexes)):
            indx = self.indexes[i]

            camera_path = self.list_camera_paths[indx]
            calib_path = self.list_lidar_paths[indx]
            label_path = self.list_label_paths[indx]
            lidar_path = self.list_calib_paths[indx]

            rot, tr, sc, image_translate_x, image_translate_y, image_translate_z, ang, fliplr = self.get_augmentation_parameters_aligned()
                    
            data_reader_obj = DataReader(camera_path, calib_path, label_path, lidar_path, rot, sc, tr, ang, image_translate_x, image_translate_y, image_translate_z, fliplr=fliplr)

            camera_image = data_reader_obj.read_image()
            lidar_image = data_reader_obj.lidar_reader.read_lidar()

                    # if self.augment:
                    #     if np.random.random_sample() >= 0.3:
                    #         camera_image = self.apply_mask(camera_image)

            _, label, truncated, occlusion = data_reader_obj.label_reader.read_label()
            label = get_target(label, truncated, occlusion, anchors=anchors)

            lidar_images[i, :, :, :] = lidar_image
            labels[i, :, :, :, :] = label
                     
                    # yield(camera_image, lidar_image, label)
        yield lidar_images, labels



    def _init_generator(self, random_split = False):
        

        self.list_camera_paths = glob.glob(os.path.join(self.base_path, 'image_2/*'))
        self.list_lidar_paths  = glob.glob(os.path.join(self.base_path, 'velodyne/*'))
        self.list_label_paths  = glob.glob(os.path.join(self.base_path, 'label_2/*'))
        self.list_calib_paths  = glob.glob(os.path.join(self.base_path, 'calib/*'))

        return self.__data_generator(self.base_path, 
                                    image_size=self.params['image_size'],
                                    lidar_size=self.params['lidar_size'], 
                                    anchors=self.params['anchors'],
                                    list_camera_paths=self.list_camera_paths[:], 
                                    list_lidar_paths=self.list_lidar_paths[:], 
                                    list_label_paths=self.list_label_paths[:], 
                                    list_calib_paths=self.list_calib_paths[:], 
                                    training=self.training)

        

    def get_next(self, batch_size=1):
        camera_tensors = []
        lidar_tensors = []
        label_tensors = []

        for _ in range(batch_size):
            camera_tensor, lidar_tensor, label_tensor = list(next(self.generator))
            camera_tensors.append(camera_tensor)
            lidar_tensors.append(lidar_tensor)
            label_tensors.append(label_tensor)

        camera_tensors = np.array(camera_tensors)
        lidar_tensors = np.array(lidar_tensors)
        label_tensors = np.array(label_tensors)

        return (camera_tensors, lidar_tensors, label_tensors)


    def apply_mask(self, image, size=30, n_squares=3):
        h, w, channels = image.shape
        new_image = image[:]
        for _ in range(n_squares):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = np.clip(y - size // 2, 0, h)
            y2 = np.clip(y + size // 2, 0, h)
            x1 = np.clip(x - size // 2, 0, w)
            x2 = np.clip(x + size // 2, 0, w)
            new_image[y1:y2,x1:x2,:] = 0
        return new_image


    def apply_mask_lidar(self, image, size_x=5, size_y=5, size_z=5, n_squares=5):
        w, l, h = image.shape
        new_image = image[:]
        for _ in range(n_squares):
            x = np.random.randint(w)
            y = np.random.randint(l)
            z = np.random.randint(h)

            y1 = np.clip(y - size_y // 2, 0, l)
            y2 = np.clip(y + size_y // 2, 0, l)
            x1 = np.clip(x - size_x // 2, 0, w)
            x2 = np.clip(x + size_x // 2, 0, w)
            z1 = np.clip(z - size_z // 2, 0, h)
            z2 = np.clip(z + size_z // 2, 0, h)
            new_image[x1:x2, y1:y2,z1:z2] = 0.
        return new_image


    def get_augmentation_parameters(self):
        if self.augment:

                    if np.random.random_sample() >= 0.0:
                        image_translate_x = random.randint(-50, 50)
                    else:
                        image_translate_x = 0
                    if np.random.random_sample() >= 0.0:
                        image_translate_y = random.randint(-25, 25)
                    else:
                        image_translate_y = 0

                    if np.random.random_sample() >= 0.0:
                        translate_x = np.random.random_sample() * 20 - 10
                    else:
                        translate_x = 0
                    if np.random.random_sample() >= 0.0:
                        # translate_y = random.randint(-15, 15)
                        translate_y = np.random.random_sample() * 20 - 10
                    else:
                        translate_y = 0

                    if np.random.random_sample() >= 0.0:
                        translate_z = random.random() - 0.5
                    else:
                        translate_z = 0

                    if np.random.random_sample() >= 0.0:
                        ang = np.random.random_sample() * 30 - 15
                    else:
                        ang = 0

                    r = R.from_rotvec(np.radians(ang) * np.array([0, 0, 1]))
                    rot = r.as_dcm()
                    rot = np.append(rot, np.array([[0,0,0]]), axis=0)
                    rot = np.append(rot, np.array([[0],[0],[0],[1]]), axis=1)

                    tr_x = translate_x
                    tr_y = translate_y
                    tr_z = translate_z
                    tr = np.array([[tr_x], [tr_y], [tr_z], [0]])
                    
                    translate_x = 0
                    translate_y = 0
                    translate_z = 0
                    
                    sc_x = 1
                    sc_y = 1
                    sc_z = 1   

                    if np.random.random_sample() >= 0.0:
                       sc_x += ((random.random() * 2) - 1.) / 10.

                    if np.random.random_sample() >= 0.0:
                       sc_y  += ((random.random() * 2) - 1.) / 10.
                    

                    sc = np.array([[sc_x, 0, 0, 0], [0, sc_y, 0, 0], [0, 0, sc_z, 0], [0, 0, 0, 1]])

                    fliplr = np.random.random_sample() >= 0.5

        else:
                    image_translate_x = 0
                    image_translate_y = 0

                    translate_x = 0
                    translate_y = 0
                    translate_z = 0
                    ang = 0

                    r = R.from_rotvec(np.radians(0) * np.array([0, 0, 1]))
                    rot = r.as_dcm()
                    rot = np.append(rot, np.array([[0,0,0]]), axis=0)
                    rot = np.append(rot, np.array([[0],[0],[0],[1]]), axis=1)

                    tr_x = 0
                    tr_y = 0
                    tr_z = 0
                    tr = np.array([[tr_x], [tr_y], [tr_z], [0]])

                    sc_x = 1
                    sc_y = 1
                    sc_z = 1
                    sc = np.array([[sc_x, 0, 0, 0], [0, sc_y, 0, 0], [0, 0, sc_z, 0], [0, 0, 0, 1]])

                    fliplr = False

        return rot, tr, sc, image_translate_x, image_translate_y, ang, fliplr






           


