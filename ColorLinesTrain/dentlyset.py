"""
Created on Wed Jan 16 04:44:24 2019

@author: alontetro
"""

import numpy as np
import glob
from random import randint
import cv2
import os

#import scipy.io
#import json

def get_images_and_labels_path(scans_list_file, 
                               indexing_ext='_indRef', 
                               segmentation_ext='_indSeg', 
                               input_format="png"):

    src_inputs_path = []
    
    for n in scans_list_file:
        i_root_data_path = n                       
        print("load data from {0}".format(i_root_data_path))
        src_inputs_path = src_inputs_path + glob.glob('{0}*[!{1}{2}].{3}'.format(i_root_data_path, indexing_ext,segmentation_ext, input_format))
        print("len(src_inputs_path)", len(src_inputs_path))
    return src_inputs_path


def load_image(mat_file_path,  mode=0):
#    return np.array(scipy.io.loadmat(mat_file_path)[pattern[mode]])
    im_out = cv2.imread(mat_file_path)
    return im_out


class ImgAugmantation:
    def __init__(self, AugmantationConfig):
        self.use_flip = AugmantationConfig["use_flip"]
        self.use_random_shifting = AugmantationConfig["use_random_shifting"]
        self.use_down_up = AugmantationConfig["use_down_up"]
        self.use_random_noise = AugmantationConfig["use_random_noise"]
        self.use_increase_brightness = AugmantationConfig["use_increase_brightness"]
        self.Frequancy = AugmantationConfig["Frequancy"]
        
    def make_random_aug(self, src, idx, seg):
        running_func = []
        if self.use_flip:
            if randint(0, 10) < self.Frequancy:
                src, idx, seg = self.aug_flip(src, idx, seg)
                running_func.append("aug_flip")
                
        if self.use_random_shifting:
            if randint(0, 10) < self.Frequancy:
                src, idx, seg = self.random_shifting(src, idx, seg)
                running_func.append("random_shifting")
                
        if self.use_down_up:
            if randint(0, 10) < self.Frequancy:
                src = self.down_up(src)
                running_func.append("down_up")
                
        if self.use_random_noise:
            if randint(0, 10) < self.Frequancy:
                src = self.random_noise(src)
                running_func.append("random_noise")
                
        if self.use_increase_brightness:
            if randint(0, 10) < self.Frequancy:
                src = self.increase_brightness(src)
                running_func.append("increase_brightness")
                
                
        return src, idx, seg, running_func
    
    def aug_flip(self, src, idx, seg):
        src = cv2.flip(src, 1)
        
        if not idx is None:
            idx = cv2.flip(idx, 1)
            
        if not seg is None:
            seg = cv2.flip(seg, 1)
            
        return src, idx, seg
    
    def random_shifting(self, src, idx, seg):
        rows,cols = src.shape[:2]

        r1  = np.random.randint(-5,5)
        r2 = np.random.randint(-5,5)
#        M = np.float32([[1,0,np.random.randint(-100,100)],[0,1,np.random.randint(-100,100)]])
        M = np.float32([[1,0,r1],[0,1, r2]])
        src = cv2.warpAffine(src,M,(cols,rows))
        
        if not idx is None:
            idx = cv2.warpAffine(idx,M,(cols,rows))
        
        if not seg is None:
            seg = cv2.warpAffine(seg,M,(cols,rows))
        
        return  src, idx, seg
    
    def down_up(self, src):
        h, w =src.shape[:2]
        
        random_s = np.random.randint(h - 300, h)
        random_r = random_s / h
        
        src = cv2.resize(src, (int(random_r*w), int(random_r*h)))      
        src = cv2.resize(src, (w, h))
        return src
    
    def random_noise(self, src):
        mean = 0.0   # some constant
        std = 1.0    # some constant (standard deviation)
        noisy_img = src + np.random.normal(mean, std, src.shape)
        noisy_img_clipped = np.clip(noisy_img, 0, 255) 
        
        return noisy_img_clipped
    
    def increase_brightness(self, src):
        value=np.random.randint(0,3)
        hsv = cv2.cvtColor(src.astype("uint8"), cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        
        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value
        
        final_hsv = cv2.merge((h, s, v))
        src = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)
        return  src
    
    
class data_loader: 
    def __init__(self, dataset_config, datatype="TrainingScansPath"):
        self.config = dataset_config
        self.src_inputs_path = get_images_and_labels_path(self.config[datatype],
                                                          indexing_ext=self.config["indexing_ext"], 
                                                          segmentation_ext=self.config["seg_ext"], 
                                                          input_format=self.config["image_format"][1:])
        print('Loading data completed')
    
    def get_random_id(self):
        return randint(0, len(self.src_inputs_path) - 1)
    
    def get_random_sample(self):
        n_id = self.get_random_id()     
        image_format = self.config["image_format"]
        c_path = self.src_inputs_path[n_id].split(image_format)[0]
        
        src = None
        idx = None
        seg = None
        
        if not os.path.isfile(self.src_inputs_path[n_id]):
            print("{0} not exist".format(self.src_inputs_path[n_id]))
            src = np.zeros(self.config["InputShape"], dtype="uint8")
        else:
            src = load_image( c_path  + image_format)
            
        if self.config['UseIdx']:
            if not os.path.isfile(c_path + self.config["indexing_ext"] + image_format):
                print("{0} not exist".format(c_path + self.config["indexing_ext"] + image_format))
                
                idx = np.zeros(self.config["InputShape"], dtype="uint8")
            else:
                idx = load_image(c_path + self.config["indexing_ext"] + image_format) 
        
        if self.config['UseSeg']:
            if not os.path.isfile(c_path + self.config["seg_ext"] + image_format):
                print("{0} not exist".format(c_path + self.config["seg_ext"] + image_format))
                
                seg = np.zeros(self.config["InputShape"], dtype="uint8")
            else:
                seg = load_image(c_path + self.config["seg_ext"] + image_format)
        
        
        return [src, idx, seg, c_path]
    
    def create_new_mask(self, labels, mask_size):
        output = np.zeros((labels.shape[0], labels.shape[1]))
        output += labels

        mask = output.copy()
        mask[mask > 0] = 1
        mask[mask <= 0] = 0
    
        return output, mask.astype("float32")  
    
    def create_new_mask1(self, labels):
        output = np.zeros((labels.shape[0] , labels.shape[1]))
        labels[labels > 0] = 1
        output += labels

        #output[output == 0] = -100.0
    
        return output       
    # 
    def get_sample_for_train(self):# 
                     
        
        src, idx, seg, c_path = self.get_random_sample()
        if self.config['UseIdx']:
            idx = idx[:, :, 0]
            
#            idx[idx % 2 == 1] = 0
        
        if self.config['UseSeg']:
            seg = seg[:, :, 0]
            

#        cv2.imshow("idx", idx)
#        cv2.imshow("src", src)
#        cv2.waitKey(0)
#        y_src[y_src % 2 == 1] = 1
#        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
#        idx, mask = self.create_new_mask(idx.copy(), 2)
        
#        if (randint(0, 10) < 5):       
#            for f in aug_function_list:
#                if (randint(0, 10) < 3):
#                    x, y, y_src = f(x, y, y_src)  
#                    
#        idx_vol, mask1 = self.create_new_mask1(idx.copy(), 2)

        return src, idx, seg, c_path  
#all_data = data_loader("D:\TempData\AI_refInd\D__TempData__{d538f6ec-2e2d-40b8-998e-4eeacc41c3e7}/")  
#a,b,c,d  =all_data.get_sample_for_train()
#

