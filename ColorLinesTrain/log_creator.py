#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 06:20:36 2019

@author: alontetro
"""

import numpy as np
import cv2
import torch 
import random 
import csv
import TrainEval
import os

import scipy
from scipy import ndimage

def read_label(path):
    return cv2.imread(path)[:,:,0]

def read_image(path):
    return cv2.imread(path)

class Skeleton:
    def __init__(self, 
                 label, 
                 sub_label=None,
                 max_i=48,
                 min_i=2,
                 shape=640,
                 use_cuda=True):
        
        if use_cuda:
            dtype = torch.cuda.FloatTensor
            dtype_l = torch.cuda.LongTensor
        else:
            dtype = torch.FloatTensor
            dtype_l = torch.LongTensor
            
        
        indices = (label).nonzero().squeeze()
        if not len(indices) > 0:
            self.ok = False
            return None
        
        if not len(indices.shape) > 1:
            self.ok = False
            return None
        
        self.ok = True
        self.y = indices[:, 0].type(dtype_l)
        self.x = indices[:, 1].type(dtype_l)
        self.idxs = label[self.y, self.x].type(dtype_l)
        
        self.idxs[self.idxs > max_i] = 0
        self.idxs[self.idxs <= min_i] = 0
        
        self.s_list = torch.zeros((1, 64, shape)).type(dtype_l)
        self.s_list[:,self.idxs, self.x] += self.y
        self.s_list = self.s_list.type(dtype)
        
        
def padding_resize(image, label=None, sub_label=None, out_size=1280):
    out_image = np.zeros((1280, 1280, 3), dtype=np.uint8)
    out_label = np.zeros((1280, 1280), dtype=np.uint8)
    
    if not sub_label is None:
        out_sub_label = np.zeros_like(out_image)
    
    h, w = image.shape[:2]
    s_w = random.randint(0, out_size-w-1)  
    s_h = random.randint(0, out_size-h-1) 
    
    out_image[s_h:s_h+h, s_w:s_w+w] += image
    if not label is None:
        out_label[s_h:s_h+h, s_w:s_w+w] += label
    
    if not sub_label is None:
        out_sub_label[s_h:s_h+h, s_w:s_w+w] = sub_label
    else: 
        out_sub_label = np.zeros_like(out_label)
        
    
    return out_image, out_label, out_sub_label

def compute_mad(data, use_cuda=True):
#    data = torch.from_numpy(data)
    mean = data.mean() #mean
    norm_err = (data-mean)
    norm_abs_err = norm_err.abs() # make it absolute
    
    return norm_abs_err.mean().detach().cpu().numpy(), mean.detach().cpu().numpy() # take the mean

    

def compute_mad_err(x, y):
    '''
    compute mean absolute deviation
    '''
#    x = torch.from_numpy(x)
#    y = torch.from_numpy(y)
    
#    x[x <= 0] = -1
#    y[y <= 0] = -2
    
    x_mask = x > 0
    
    y_mask = y > 0
    
    xy_mask = x_mask*y_mask
    
    err = y[xy_mask] - x[xy_mask]
    
    return compute_mad(err), err.detach().cpu().numpy()

def compute_histogram(data, n_bins=None):
    if not n_bins is None:
        hist, bins= np.histogram(data, bins=n_bins)
    else:
        hist, bins= np.histogram(data, bins='auto')
    
    return hist, bins

def get_matrix_accuracy(ai, gt):
    ai = ai.astype("float32")    
    gt = gt.astype("float32")
    
    ai_mask = ai > 0 
    gt_mask = gt > 0 
    aigt_mask = np.logical_and(ai_mask, gt_mask)

    true_mask = ai[aigt_mask] == gt[aigt_mask]
    
    correct_pixels = true_mask.sum()
    p_inAI_and_inGT = aigt_mask.sum()
    wrong_pixels = p_inAI_and_inGT -  correct_pixels
    
    accuracy = correct_pixels / max(p_inAI_and_inGT, 1)
    return correct_pixels, wrong_pixels, p_inAI_and_inGT, accuracy

def duplicate_label(label):
    out = label.copy()
    temp = label.copy()
    
    out[:-1, :] += temp[1:,:]
    out[:-2, :] += temp[2:,:]
    out[:-3, :] += temp[3:,:]
    out[1:, :] += temp[:-1,:]
    out[2:, :] += temp[:-2,:]
    out[3:, :] += temp[:-3,:]
    
    return out


class test_summary:
    def __init__(self, 
                 test_data, 
                 trained_model_name, 
                 TrainingLogDir, 
                 ModelName, 
                 pyname, 
                 training_step):
        self.root_folder = '{0}/{1}/static'.format(TrainingLogDir, ModelName)
        self.test_data = test_data
        self.eval_model = TrainEval.Eval(TrainingLogDir, ModelName, pyname)
        self.eval_model.load_Weights(trained_model_name)
        self.training_step = training_step
        self.summary_dict = {}
        self.accuracyIndexAll = []
        self.accuracySkelAll = []
        
        try:
            os.makedirs(self.root_folder)
        except FileExistsError:
            print("TrainingLogDir directory already exists")
            pass
        
        
        
    def start(self, 
              max_steps_per_folder=30,
              use_cuda=True):
        
        i = 0
        accuracyIndex = 0
        accuracySkel = 0
        indexSum = 0
        for path in self.test_data.src_inputs_path[:max_steps_per_folder]:
            print(i)
            path1 = path.replace('\\', '/')
            i += 1
            c_path = path1.split(".png")[0]
            img_number = c_path.split("/")[-1][:]
            path = c_path.split("/")[-1][2:]
            if not img_number in self.summary_dict:
                self.summary_dict[img_number] = {}
                    
            real_image_orig = read_image( c_path + ".png")
            y = read_label(c_path + "_indRef.png")

            src_skeleton = y.copy()
                
            real_image = self.eval_model.img_to_input_format(real_image_orig)

            mask = src_skeleton != 0
            mask2 = ndimage.binary_dilation(mask, iterations=1300)
            real_image[0, :, mask2 == 0] = 0
            real_image_orig[mask2 == 0,:] = 0

            #final_res, skeleton_final, skeleton_pred, indexing_pred, indexing_argmax = self.eval_model.model.forward(real_image)           
            final_res, skeleton_final, skeleton_pred, indexing_tensor, indexing_argmax, indexing_pred = self.eval_model.model.forward_eval(real_image)

            with torch.no_grad():
    #            torch.cuda.synchronize()
                indexing_argmax = indexing_argmax.cpu().data.numpy()
                skeleton_pred = skeleton_pred.cpu().data.numpy()
                skeleton_final = skeleton_final.cpu().data.numpy()
                final_res = final_res.cpu().data.numpy()

            final_res = final_res[0].copy()
            skeleton_final = skeleton_final[0, 0].copy()
            skeleton = skeleton_pred[0, 0].copy()
            skeleton1 = skeleton.copy()
            skeleton2 = skeleton.copy()


            indexing_argmax = indexing_argmax[0].copy()
            indexing_argmax1 = indexing_argmax.copy()
            indexing_argmax2 = indexing_argmax.copy()

            duplicate_src = duplicate_label(src_skeleton)
            duplicate_mask = duplicate_src > 0
            idxAI_eqq_idxGT = indexing_argmax[duplicate_mask] >= 0.5 * duplicate_src[duplicate_mask]
            mask_skel = indexing_argmax > 0

            idxAI_eqq_idxGT = skeleton[duplicate_mask] >= 0.5 * duplicate_src[duplicate_mask] #### skel
            mask_skel = skeleton > 0 #### skel

            in_p_not_ds = indexing_argmax[mask_skel] != duplicate_src[mask_skel]
            in_p_not_ds = skeleton[mask_skel] != duplicate_src[mask_skel]  #### skel
                
            duplicate_mask = np.logical_not(duplicate_mask)
            mask_skel = np.logical_not(mask_skel)
            indexing_argmax1[duplicate_mask] = 0
            skeleton1[duplicate_mask] = 0 #### skel

            duplicate_src[mask_skel] = 0
            duplicate_src[duplicate_src == 255] = 0
            h, w = indexing_argmax1.shape[:2]
            pred_skl = Skeleton(label=torch.from_numpy(indexing_argmax1), shape=h, use_cuda=use_cuda)
            label_skl = Skeleton(label=torch.from_numpy(src_skeleton), shape=h, use_cuda=use_cuda)

            pred_skl = Skeleton(label=torch.from_numpy(skeleton1), shape=h, use_cuda=use_cuda) #### skel
            label_skl = Skeleton(label=torch.from_numpy(src_skeleton), shape=h, use_cuda=use_cuda) #### skel


            err_hist, err_bins = None, None
            err_hist, err_bins = None, None
            
            if label_skl.ok and pred_skl.ok: 
                (err_mad, err_mean), err = compute_mad_err(pred_skl.s_list, label_skl.s_list)
                err_hist, err_bins = compute_histogram(err, n_bins=[-1280,-3,-2,-1,0,1,2,3,1280])
            else:
                err_mad, err_mean, err = -1, -1, -1
                err_hist = [-1]
                
                
            indexing_argmax[indexing_argmax < 0] = 0
            indexing_argmax2[mask == 0] = 0
            y = y/2
            y = y % 28

            cv2.imwrite(self.root_folder + '/{}_IndexFinal.png'.format(img_number), indexing_argmax2*6)
            cv2.imwrite(self.root_folder +'/{}_Index.png'.format(img_number), indexing_argmax*6)
            cv2.imwrite(self.root_folder +'/{}_Indexlabel.png'.format(img_number), y*6)
            cv2.imwrite(self.root_folder + '/{}_RealImage.png'.format(img_number), real_image_orig)
            cv2.imwrite(self.root_folder + '/{}_ResultFinal.png'.format(img_number), final_res*6)

            #### skel:
            skeleton = (skeleton-np.min(skeleton))/(np.max(skeleton)-np.min(skeleton))*255
            skeleton_final[skeleton_final > 0.01] = 255
            skeleton_final[skeleton_final < 0.01] = 0
            #skeleton_final[skeleton_final < 50] = 0

            cv2.imwrite(self.root_folder + '/{}_skelFinal.png'.format(img_number), skeleton_final.astype(int))
            cv2.imwrite(self.root_folder +'/{}_skel.png'.format(img_number), skeleton.astype(int))
            cv2.imwrite(self.root_folder +'/{}_skellabel.png'.format(img_number), y*255)

            
            correct_pixels, wrong_pixels, tested_pixels, accuracy = get_matrix_accuracy(indexing_argmax2, src_skeleton)
        
            accuracyIndex += np.sum(indexing_argmax2[indexing_argmax2>0] == y[indexing_argmax2>0])
            accuracySkel += np.sum(skeleton_final[y>0] > 0)
            indexSum += np.sum(y>0)
                        
            
            self.summary_dict[img_number]['titles'] = ['image_path',
                                 'idxAI_eqq_idxGT', 
                                 'err_mad', 
                                 'err_mean',
                                 'correct_pixels', 
                                 'wrong_pixels', 
                                 'tested_pixels', 
                                 'accuracy', 
                                 'err_bins: ' + str([-1280,-3,-2,-1,0,1,2,3,1280])]
                
              
            self.summary_dict[img_number] = [c_path,
                                 str(idxAI_eqq_idxGT.sum())[:7], 
                                 str(err_mad)[:7], 
                                 str(err_mean)[:7], 
                                 str(correct_pixels.sum())[:7], 
                                 str(wrong_pixels)[:7], 
                                 str(tested_pixels)[:7], 
                                 str(accuracy)[:7], 
                                 str(err_hist)]
        self.accuracyIndexAll = accuracyIndex/indexSum
        self.accuracySkelAll = accuracySkel/indexSum
                  
#                r1 = src_skeleton > 0
#                ss = skeleton[r1 > 0] != src_skeleton[r1 > 0]
#                vis_indexing = self.eval_model._draw_results(src_x.copy(), skeleton)
#                real_indexing = self.eval_model._draw_results(src_x.copy(), src_skeleton)
#                
#                cv2.imwrite(c_path + "_pred.png",  vis_indexing.astype("uint8"))
#                cv2.imshow("label_indexing",  real_indexing.astype("uint8"))
#                cv2.waitKey(0)

            
            
    def write_sammary(self):                
        with open(self.root_folder + '/static_{0}.csv'.format(self.training_step), 'w', newline='') as csvfile:    
            t_str = 'err_bins: ' + str([-1280,-3,-2,-1,0,1,2,3,1280])
            fieldnames = ['image_path', 
                              'idxAI_eqq_idxGT',
                              'err_mad', 
                              'err_mean',
                              'correct_pixels', 
                              'wrong_pixels', 
                              'tested_pixels', 
                              'accuracy', 
                              t_str]
                
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
                
            for img_name in self.summary_dict:  
                if not img_name == 'titles':                        
                    self.summary_dict[img_name][0]
                    writer.writerow({'image_path': self.summary_dict[img_name][0], 
                                     'idxAI_eqq_idxGT': self.summary_dict[img_name][1], 
                                     'err_mad': self.summary_dict[img_name][2], 
                                     'err_mean': self.summary_dict[img_name][3], 
                                     'correct_pixels': self.summary_dict[img_name][4], 
                                     'wrong_pixels': self.summary_dict[img_name][5], 
                                     'tested_pixels': self.summary_dict[img_name][6], 
                                     'accuracy': self.summary_dict[img_name][7], 
                                     t_str: self.summary_dict[img_name][8],})
                

            

            
        
        
        
        
#def write_summary_
#
#with open('./test_log/employee_file.csv', mode='w') as employee_file:
#    employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#
#    employee_writer.writerow(['John Smith', 'Accounting', 'November'])
#    employee_writer.writerow(['Erica Meyers', 'IT', 'March'])
#    
#    
#    
    





#a = Dently_eval.Eval()
#a.load_Weights('/Users/alontetro/Downloads/1.9.19/new_files/traind_model_115200.pt')
#example_config = {"test_folders_list": ['C:/Users/User/Desktop/alon_2019/UI_dentlytech/test_data_1280/'],
#                  "weights": 'C:/Users/User/Desktop/alon_2019/UI_dentlytech/1.10.19/log2/traind_model_32000.pt',
#                  "log_folder": 'C:/Users/User/Desktop/alon_2019/UI_dentlytech/log_new/'}
#
#
#p = test_summary(example_config)
#p.start()
#p.write_sammary()
#real_image, skeleton_o, indexing_output, indexing_argmax, seg_output, seg_argmax, sk_x, s1, s2   = a.from_path(p)  
#cv2.imshow("real_image",  real_image.astype("uint8"))
#cv2.waitKey(0)
#test_label_path = '/Users/alontetro/Downloads/AI_Labels/00000086_min_indRef.png'
#y_src = read_label(test_label_path)
#temp1 = np.zeros((1280,1280))
#temp1[s1:s1 + 640, s2:s2+640] += y_src.copy()

#all_data = dentlyset.data_loader(['/Users/alontetro/Downloads/AI_Labels/'])  
##
#x, y_src, y_src, yy, c_path, dk_l =all_data.get_sample_for_train()

#vis_indexing = a._draw_results(real_image, skeleton_o[0])
#real_indexing = a._draw_results(real_image, temp1)
#                            
#cv2.imshow("pred_indexing",  vis_indexing.astype("uint8"))
#cv2.imshow("label_indexing",  real_indexing.astype("uint8"))
#cv2.waitKey(0)


#image = read_image('/Users/alontetro/Downloads/D__TempData__70ea14c0-9a84-4cd2-8ee9-ba4edf69f6a7_extGT/00000056_min_indRef.png')
#
#aa, bb, cc = padding_resize(image)
#torch_image = torch.from_numpy(aa)
#
##import cv2 
##cv2.imshow("aa", aa)
##cv2.imshow("bb", bb)
##cv2.waitKey(0)
## is the same as
##a.index(4)
## is the same as
#temp_dict = {}
#a = time.time()

#print(compute_mad(sk_o.s_list))
#print(compute_mad(sk_o1.s_list))
#
#hist, bins = compute_histogram(sk_o.idxs)
#
#
#plt.bar(bins[:-1], hist, width = 1)
#plt.xlim(min(bins), max(bins))
#plt.show()   
#
#hist, bins = compute_histogram(sk_o1.idxs)
#
#
#plt.bar(bins[:-1], hist, width = 1)
#plt.xlim(min(bins), max(bins))
#plt.show()   
#import matplotlib.pyplot as plt
#import numpy as np
#counter, bins = compute_histogram(sk_o.idxs)
##plt.hist(sk_o.i, bins=50)
##plt.ylabel('Probability');
#print(time.time() - a)
#    
    
#temp = temp.transpose(0, 1)
#temp = torch_image[temp]