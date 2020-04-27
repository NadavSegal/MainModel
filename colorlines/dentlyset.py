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
import scipy.io
import json

def get_images_and_labels_path(train_teeth_list_file, 
                               labels_pattern='_indRef', 
                               file_e="png"):

    src_inputs_path = []
    src_labels_path = []
    
    for n in train_teeth_list_file:
        i_root_data_path = n                       
        print("load data from {0}".format(i_root_data_path))
        src_inputs_path = src_inputs_path + glob.glob('{0}*[!{1}].{2}'.format(i_root_data_path, labels_pattern, file_e))
        src_labels_path = src_labels_path + glob.glob('{0}*{1}.{2}'.format(i_root_data_path, labels_pattern, file_e))
        print("len(src_inputs_path)", len(src_inputs_path))
    return src_inputs_path, src_labels_path


#def get_images_and_labels_path(train_teeth_list_file='/media/alon/SP PHD U3/AI_Ref/F__Records__ProtoTypeV2__Invitro__S_12.6.2__Tooth10_WithoutBlood_z/', 
#                               labels_pattern='_indRef', 
#                               file_e="png"):
#
#    src_inputs_path = []
#    src_labels_path = []
##    
##    with open(train_teeth_list_file) as json_file:  
##        data = json.load(json_file)
##        for n in data.keys():
#    i_root_data_path = '/home/alon/dently_7_19/F__Records__ProtoTypeV2__Invitro__S_12.6.2__Tooth10_WithoutBlood_z/'                       
#    print("load data from {0}".format(i_root_data_path))
#    src_inputs_path =  glob.glob('{0}*[!{1}].{2}'.format(i_root_data_path, labels_pattern, file_e))
#    src_labels_path =  glob.glob('{0}*{1}.{2}'.format(i_root_data_path, labels_pattern, file_e))
#        
#    return src_inputs_path, src_labels_path

    
def load_mat_file_as_np(mat_file_path, pattern=['img','label'], mode=0):
#    return np.array(scipy.io.loadmat(mat_file_path)[pattern[mode]])
    im_out = cv2.imread(mat_file_path)
    return im_out
    
class data_loader: 
    def __init__(self, train_teeth_list_file='/home/alon/dently_7_19/F__Records__ProtoTypeV2__Invitro__S_12.6.2__Tooth10_WithoutBlood_z/'):
        self.src_inputs_path, self.src_labels_path = get_images_and_labels_path(train_teeth_list_file)
        self.rp = train_teeth_list_file
        assert len(self.src_inputs_path) == len(self.src_labels_path)
        print('Loading data completed')
    
    def get_random_id(self):
        return randint(0, len(self.src_inputs_path) - 1)
    
    def get_random_sample(self):
        n_id = self.get_random_id()        
        c_path = self.src_inputs_path[n_id].split(".png")[0]
#        print(c_path)
        return [load_mat_file_as_np( c_path + ".png"), load_mat_file_as_np(c_path + "_indRef.png", mode=1), c_path]
    
    def create_new_mask(self, labels, mask_size):
        output = np.zeros((labels.shape[0], labels.shape[1]))
        
#        output[int(mask_size/2):-1*int(mask_size/2)] += labels
        output += labels
#        
#        for i in range(1, mask_size):       
#            mask = output[i:i-mask_size] == 0
#            output[i:i-mask_size][mask] += labels[mask]
#     
#        output = output[int(mask_size/2):-1*int(mask_size/2)]
    
        mask = output.copy()
        mask[mask > 0] = 1
        mask[mask <= 0] = 0
    
        return output, mask.astype("float32")  
    
    def create_new_mask1(self, labels, mask_size):
        output = np.zeros((labels.shape[0] , labels.shape[1]))
        labels[labels > 0] = 1
        output += labels
        
#        for i in range(1, mask_size):       
#            mask = output[i:i-mask_size] == 0
#            output[i:i-mask_size][mask] += (labels[mask]*(-50))
     
#        output = output[int(mask_size/2):-1*int(mask_size/2)]
        output[output == 0] = -100.0
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

        M = np.float32([[1,0,np.random.randint(-50,50)],[0,1,np.random.randint(-50,50)]])
        x = cv2.warpAffine(x,M,(cols,rows))
        y = cv2.warpAffine(y,M,(cols,rows))
        y_src = cv2.warpAffine(y_src,M,(cols,rows))
        
        return  x, y, y_src
    
    def down_up(x, y, y_src):
            src_size =x.shape[:2]
            s = np.random.randint(320, 580)
            x = cv2.resize(x, (s, s))
#            y = cv2.resize(y, (s, s), cv2.INTER_NEAREST)
            
            x = cv2.resize(x, src_size)
#            y = cv2.resize(y, src_size, cv2.INTER_NEAREST)
#            t, b, l, r = int(s/2) -128, int(s/2) + 128, int(s/2) -128, int(s/2) + 128  
            return x, y, y_src
    
    def random_noise(x, y, y_src):
        mean = 0.0   # some constant
        std = 1.0    # some constant (standard deviation)
        noisy_img = x + np.random.normal(mean, std, x.shape)
        noisy_img_clipped = np.clip(noisy_img, 0, 255) 
        
        return noisy_img_clipped, y, y_src
    
    def increase_brightness(x, y, y_src):
        value=np.random.randint(0,3)
        hsv = cv2.cvtColor(x.astype("uint8"), cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        
        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value
        
        final_hsv = cv2.merge((h, s, v))
        x = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)
        return  x, y, y_src
        
    # 
    def get_sample_for_train(self, aug_function_list=[aug_flip , increase_brightness, random_shifting, down_up]):# 
                     
        
        x, y_src, c_path = self.get_random_sample()
        y_src = y_src[:, :, 0]
#        y_src[y_src % 2 == 1] = 1
#        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        y, mask = self.create_new_mask(y_src.copy(), 2)
        yy, mask1 = self.create_new_mask1(y_src.copy(), 2)
        
        if (randint(0, 10) < 5):       
            for f in aug_function_list:
                if (randint(0, 10) < 3):
                    x, y, y_src = f(x, y, y_src)  

        return x, y_src.copy(), y_src, yy, c_path
    
#all_data = data_loader("D:\TempData\AI_refInd\D__TempData__{d538f6ec-2e2d-40b8-998e-4eeacc41c3e7}/")  
#a,b,c,d  =all_data.get_sample_for_train()
#
#cv2.imshow("aa", d)
#cv2.waitKey(0)
