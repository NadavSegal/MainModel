import numpy as np
import time

import torch
import dentlyset 
import torch.nn.functional as F
import cv2 
from losses import dont_care_crossentropy
from losses import skeleton_based_loss
import utils   
import dently_model as dently_model
import Dently_eval
import json_manager
    
    
def postprocces_pred(output, arg_max):
    with torch.no_grad():
        torch.cuda.synchronize()
        aa = torch.squeeze(output, 0)
        a = aa.cpu().data.numpy()
        torch.cuda.synchronize()
        bb = torch.squeeze(arg_max, 0)
        b = bb.cpu().data.numpy()
        torch.cuda.synchronize()
            
    return  a, b 


class Train:
    def __init__(self, config_path="./config_data/model_config.json"):
        '''
        path_to_trained_model: path to trainrd pt file for example: ./traind_model_316000.pt'
        '''
        
        self.model_config = json_manager.read_dict_from_json(config_path)
                                    
        self.model_config['use_cuda'] = torch.cuda.is_available()
        self.model  = dently_model.DentlyNet()
        self.adamOptimizer = torch.optim.Adam(self.model.parameters(), lr=self.model_config['learning_rate'])
        self.clasification_loss = dont_care_crossentropy()
        self.skeleton_loss = skeleton_based_loss()
        
        if self.model_config['last_trained_model_path']:
            self.load_Weights(self.model_config['last_trained_model_path'])
           
        self.model.train()
        
        if self.model_config['use_cuda']:
            self.model.cuda()
                
                
        print("--- loading training data ---")    
        self.train_dentlyset =  dentlyset.data_loader(self.model_config['training_data_path'])
        print("--- finished ---")
        
        print("--- loading training data ---")    
        self.test_dentlyset =  dentlyset.data_loader(self.model_config['test_data_path'])
        print("--- finished ---")
        
        
    def load_Weights(self, pt_trained_model_path=None, ad_trained_model_path=None):
        '''
        pt_trained_model_path: path to trainrd pt file for example: ./traind_model_316000.pt'
        '''
        if pt_trained_model_path:
            print("loading from trained model path: {0}".format(pt_trained_model_path))
            
            if self.model_config['use_cuda'] :
                
                model_dict = self.model.state_dict()
                self.model.load_state_dict(model_dict)
                pretrained_dict = torch.load(pt_trained_model_path) 

                # 1. filter out unnecessary keys
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}# and not "conv_59_32_32_output" in k.split(".") and not "ce3" in k.split(".")}
                # 2. overwrite entries in the existing state dict
                model_dict.update(pretrained_dict) 

                self.model.load_state_dict(model_dict)
                try:
                    self.model.cuda()
                except:
                    print("Can't load the model on gpu")
                    print("using cpu mode")
            else:
                self.model.load_state_dict(torch.load(pt_trained_model_path, map_location={'cuda:0': 'cpu'}))
        else:
            print("Please check if the model get the correct pt file path")
    
    def save_model(self, global_step, test_image=None):
        torch.save(self.model.state_dict(), '{0}/traind_model_{1}.pt'.format(self.model_config['log_folder_path'], global_step)) 
        
        return  '{0}/traind_model_{1}.pt'.format(self.model_config['log_folder_path'], global_step)
    
    
    def start(self):        
        global_step = 0
        while(global_step < self.model_config['training_max_steps']):
            loss = None
            i = 0
            max_y = 0
            min_y = 0
            while i < self.model_config['training_batch_size']:
                x1, y1, y_src1, yy, c_path = self.train_dentlyset.get_sample_for_train()
#                show_img = x1.copy()
                max_y = y1.max()
                min_y = y1.min()
                if max_y < 1 or max_y > self.model_config['indexing_max'] or min_y < 0:
                    print("look at {0}".format(c_path))
                    continue
                    
                tt1 = y1.copy()        
                x1, y_t1 = utils.img_and_labels_to_model_input_format(x1, y1)
                
                    
                d_pred, seg_pred = self.model(x1)
                
                pred_after_softmax = torch.exp(F.log_softmax(d_pred, 1))
                output, arg_max = pred_after_softmax.max(dim=1)
                output = torch.where((arg_max != 0), output, torch.zeros(output.size(0), output.size(1), output.size(2)).type(self.model.dtype) - 50)
                
                if loss is None:
                    loss2 = self.skeleton_loss(output, yy) #
                    loss = self.clasification_loss(d_pred, y_t1, tt1) + loss2  #+ l1
                            
                else:
                    loss2 = self.skeleton_loss(output, yy)
                    loss += self.clasification_loss(d_pred, y_t1, tt1) + loss2  #+ l1
                    
                if (global_step % self.model_config['test_after_N'] == 0):
                    current_trained_path = self.save_model(global_step)
                    
                    test_model = Dently_eval.Eval()
                    test_model.load_Weights(current_trained_path)
                    
                    for test_i in range(self.model_config['test_steps']):
                        x1, y1, y_src1, yy, c_path = self.test_dentlyset.get_sample_for_train()
                        test_img_path = c_path + ".png"
                        test_label_path = c_path + '_indRef.png'
                        y_src = cv2.imread(test_label_path)
                        y_src = y_src[:, :, 0]
    
                        real_image, skeleton, indexing_output, indexing_argmax, seg_output, seg_argmax = test_model.from_path(test_img_path)

                        p1, n1 = utils.get_matrix_accuracy(skeleton[0].copy(), y_src.copy())
                        print("acc1  = {0}, {1}".format(p1, n1))
                        
                        if self.model_config['vis_training']:
                            vis_indexing = test_model._draw_results(real_image.copy(), skeleton[0])
                            real_indexing = test_model._draw_results(real_image.copy(), y_src)
                            
                            cv2.imshow("pred_indexing",  vis_indexing.astype("uint8"))
                            cv2.imshow("label_indexing",  real_indexing.astype("uint8"))
                            cv2.waitKey(100)
                            
                           

#                        
                global_step += 1 
                i += 1

            loss = loss/ self.model_config['training_batch_size']
            self.adamOptimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.adamOptimizer.step()    
            

if __name__ == '__main__':
     a = Train()
     a.start()





