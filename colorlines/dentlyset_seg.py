#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 04:44:24 2019

@author: alontetro
"""

import numpy as np
import glob
from random import randint
import cv2
# import scipy.io
# import h5py 

def get_images_and_labels_path(i_root_data_path='/home/alon/Documents/dentlytech_new/work_period_dentlytech/fullset', 
                               labels_pattern='gt', 
                               file_e="mat"):
        
    print("load data from {0}".format(i_root_data_path))
    src_inputs_path = glob.glob('{0}*[!{1}].{2}'.format(i_root_data_path, labels_pattern, file_e))
    src_labels_path = glob.glob('{0}*{1}.{2}'.format(i_root_data_path, labels_pattern, file_e))
        
    return src_inputs_path, src_labels_path
    
def load_mat_file_as_np(mat_file_path, pattern=['img','label'], mode=0):
    return np.array(scipy.io.loadmat(mat_file_path)[pattern[mode]])

def load_seg_file(seg_file_path='/Users/alontetro/Documents/dentlytech_alon/1.2_segmentation/dataForAlon.mat'):
    seg_data = h5py.File(seg_file_path)
    images = seg_data['data']['image']
    masks = seg_data['data']['mask']
    
    return seg_data, images, masks

class data_loader: 
    def __init__(self, i_root_data_path):
        self.seg_data, self.images, self.masks = load_seg_file(i_root_data_path)
        print('Loading data completed')
    
    def get_random_id(self):
        return randint(0, 4262)
    
#    def get_random_sample(self):
#        n_id = self.get_random_id()
#        c_path = self.src_inputs_path[n_id].split(".mat")[0]
#        return [load_mat_file_as_np(c_path + ".mat"), load_mat_file_as_np(c_path + "_gt.mat", mode=1)]
    
    def create_new_mask(self, labels, mask_size):
        output = np.zeros((labels.shape[0] + mask_size, labels.shape[1]))
        
        output[int(mask_size/2):-1*int(mask_size/2)] += labels
        
        for i in range(1, mask_size):       
            mask = output[i:i-mask_size] == 0
            output[i:i-mask_size][mask] += labels[mask]
     
        output = output[int(mask_size/2):-1*int(mask_size/2)]
    
        mask = output.copy()
        mask[mask > 0] = 1
        mask[mask <= 0] = 0
    
        return output, mask.astype("float32")  
    
    def aug_flip(x, y, y_src):
            x = cv2.flip(x, 1)
            y = cv2.flip(y, 1)
            y_src = cv2.flip(y_src, 1)
            return x, y, y_src
        
        
    def random_shifting(x, y, y_src):
        rows,cols = x.shape[:2]

        M = np.float32([[1,0,np.random.randint(-320,320)],[0,1,np.random.randint(-320,320)]])
        x = cv2.warpAffine(x,M,(cols,rows))
        y = cv2.warpAffine(y,M,(cols,rows))
        y_src = cv2.warpAffine(y_src,M,(cols,rows))
        
        return  x, y, y_src
    
    def random_noise(x, y, y_src):
        mean = 0.0   # some constant
        std = 1.0    # some constant (standard deviation)
        noisy_img = x + np.random.normal(mean, std, x.shape)
        noisy_img_clipped = np.clip(noisy_img, 0, 255) 
        
        return noisy_img_clipped, y, y_src
    
    def increase_brightness(x, y, y_src):
        value=np.random.randint(20,65)
        hsv = cv2.cvtColor(x.astype("uint8"), cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        
        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value
        
        final_hsv = cv2.merge((h, s, v))
        x = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)
        return  x, y, y_src
    
    def down_up(x, y, y_src):
            src_size =x.shape[:2]
            s = np.random.randint(160, 240)
            x = cv2.resize(x, (s, s))
#            y = cv2.resize(y, (s, s), cv2.INTER_NEAREST)
            
            x = cv2.resize(x, src_size)
#            y = cv2.resize(y, src_size, cv2.INTER_NEAREST)
#            t, b, l, r = int(s/2) -128, int(s/2) + 128, int(s/2) -128, int(s/2) + 128  
            return x, y, y_src
#    
#    def get_sample_for_train(self, aug_function_list=[aug_flip]):
#                     
#        
#        x, y_src = self.get_random_sample()
#        
#        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
#        y, mask = self.create_new_mask(y_src.copy(), 4)
#        
#        for f in aug_function_list:
#            if (randint(0, 10) < 5):
#                x, y, y_src = f(x, y, y_src)  
#
#        return x, y.astype('uint8'), y_src
# 
    def get_seg_sample_for_train(self, aug_function_list=[aug_flip, random_shifting, random_noise, increase_brightness, down_up]):
        random_id = self.get_random_id()
        
        image = np.array(self.seg_data[self.images[random_id][0]])
        mask = np.array(self.seg_data[self.masks[random_id][0]])
        
        image_src = image.transpose((2, 1, 0))
        image_src = cv2.cvtColor(image_src, cv2.COLOR_BGR2RGB)
        mask_src = mask.transpose((1, 0))
        
        y, mask = self.create_new_mask(mask_src.copy(), 8)
        
        
        for f in aug_function_list:
            if (randint(0, 10) < 7):
                image_src, y, mask_src = f(image_src, y, mask_src)  
        
        return image_src, y.astype('uint8'), mask_src
    
#a  = data_loader('/Users/alontetro/Documents/dentlytech_alon/1.2_segmentation/dataForAlon.mat')
#image, mask = a.get_seg_sample_for_train()
#
#image = image.transpose((2, 1, 0))
#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#mask = mask.transpose((1, 0))
#cv2.imshow("image", image.astype("uint8"))
#cv2.imshow("mask", mask.astype("uint8")*80)
#cv2.waitKey(0)
#a,b,c  =all_data.get_next_train_batch()
#
#cv2.imshow("aa", c*255+b)
#cv2.waitKey(0)
