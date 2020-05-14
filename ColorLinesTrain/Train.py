import TrainConfig 
import TrainUtils
import dentlyset
from log_creator import test_summary
from tqdm import tqdm

import scipy
from scipy import ndimage
import numpy as np
import torch.nn as nn


class Train:
    def __init__(self):        
        TrainConfig.CreateIferenceDir()
        self.train_config = TrainConfig.train_config
        self.model_config = self.train_config["ModelConfig"]
        
        self.train_config['use_cuda'] = TrainUtils.CheckCuda()
        if self.train_config['use_cuda']:
            print("Cuda is available :)")
        else:
            print("Only cpu is available :)")
           
        self.model  = TrainUtils.importModel(self.train_config['TrainingLogDir'], self.model_config["ModelName"], "model.py")
        self.optimizer = TrainUtils.getOptimizer(self.model.parameters(), self.train_config['learningRate'])
        self.clasification_loss = TrainUtils.dont_care_crossentropy()
        self.skeleton_loss = TrainUtils.skeleton_based_loss(self.train_config['use_cuda'])
#        
        if self.train_config["RetrainModel"]:
            pt_full_path = self.train_config["RetrainModel_fullpath"]
            self.model = TrainUtils.loadWeights(self.model, pt_full_path, self.train_config['use_cuda'])
   
        self.model.train()
        

        ###### set trainable to false
        if 0:#not self.model_config["UseIdx"]:
            ct = 0
            for child in self.model.children():
                ct += 1
                if ct < 28:
                    print(child)
                    for param in child.parameters():
                        param.requires_grad = False

        if 0:#not self.model_config["UseSkel"]:
            ct = 0
            for child in self.model.children():
                ct += 1
                if ct >= 28:
                    print(child)
                    for param in child.parameters():
                        param.requires_grad = False



        if self.train_config['use_cuda']:
            self.model.cuda()
            
            
        self.add_augmentation()
        self.model.half()
        self.batchnorm_to_fp32()
    
    def batchnorm_to_fp32(self):
        for layer in self.model.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.float()
    
    
    def add_augmentation(self):
        self.augmantation = dentlyset.ImgAugmantation(self.train_config["AugmantationConfig"])
            
            
    def LoadData(self):           
        print("--- loading training data ---")    
        self.train_dentlyset =  dentlyset.data_loader(self.train_config["DatasetConfig"], "TrainingScansPath")
        print("--- finished ---")
        self.train_counter = 0
            
        print("--- loading training data ---")    
        self.test_dentlyset =  dentlyset.data_loader(self.train_config["DatasetConfig"], "TestingScansPath")
        self.test_counter = 0
        print("--- finished ---")
        
    def GetWeights(self):
        print("calc Weights")
        Weights = np.zeros(28)   
        for j in range(1):
            _, idx, _, _ = self.train_dentlyset.get_sample_for_train()
            y = idx/2
            #y = y % 28        
            for i in range(1,28+1):
                Weights[i-1] = Weights[i-1] + np.sum(y==i)
        
        Weights = 1/Weights*np.sum(Weights)
        Weights[np.isinf(Weights)] = 0
        self.Weights = Weights
        
        

    
    def forward_step(self, srcModelInput):        
        #indexing_most_likely, indexing_pred, indexing_argmax = self.model(srcModelInput)
        #final_res, skeleton_final, skeleton_pred, indexing_pred, indexing_argmax = self.model(srcModelInput)
        final_res, skeleton_final, skeleton_pred, indexing_tensor, indexing_argmax, indexing_pred = self.model(srcModelInput)
        return final_res, skeleton_final, skeleton_pred, indexing_tensor, indexing_argmax
    
    def backward_step(self, loss):      
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()    
    
    def train_step(self):
        src, idx, seg, c_path = self.train_dentlyset.get_sample_for_train()
        src, idx, seg, running_func = self.augmantation.make_random_aug(src, idx, seg)
        #mask = idx != 0
        #mask = ndimage.binary_dilation(mask, iterations = 1300)
        #src[mask == 0, :] = 0
        srcModelInput = src.copy()
        srcModelInput = TrainUtils.to_torch_format(srcModelInput, self.train_config['use_cuda'])
        #indexing_most_likely, idx_per_channel_scores, skeleton_scores = self.forward_step(srcModelInput)
        final_res, skeleton_final, skeleton_pred, indexing_pred, indexing_argmax = self.forward_step(srcModelInput)

        idx_loss = 0
        if self.model_config["UseIdx"]:
            idxModelInput = idx.copy()
            idxModelInput = TrainUtils.to_torch_format(idxModelInput, self.train_config['use_cuda'])
            idx_loss = self.compute_clasification_loss(indexing_pred, idxModelInput, idx)

        skel_loss = 0
        if self.model_config["UseSkel"]:
            skelModelInput = idx.copy()
            skelModelInput = self.train_dentlyset.create_new_mask1(skelModelInput)
            
            skelModelInput1 = idx.copy()
            skelModelInput1 = TrainUtils.to_torch_format(skelModelInput1, self.train_config['use_cuda'])   
            #output, arg_max = TrainUtils.skel_score_postprocess(idx_per_channel_scores, self.train_config['use_cuda'])
            #skel_loss = self.compute_skeleton_loss(output, arg_max, skelModelInput, skelModelInput1)
            skel_loss = self.compute_skeleton_lossNEW(skeleton_pred, skelModelInput, skelModelInput1)
        
        return idx_loss, skel_loss

    
    def compute_skeleton_loss(self, output, arg_max, yy, y_t1):      
        skeleton_loss = self.skeleton_loss(output, yy, arg_max,  y_t1) 
        
        return skeleton_loss

    def compute_skeleton_lossNEW(self, skeleton_scores, yy, y_t1):
        skeleton_loss = self.skeleton_loss(skeleton_scores, yy, y_t1)

        return skeleton_loss

    def compute_clasification_loss(self, per_channel_scores, ModelInput, src):
        clasification_loss = self.clasification_loss(per_channel_scores, ModelInput, src)

        return clasification_loss
     
import time        
def mainTrainLoop():
    train_new = Train()
    train_new.LoadData()
    train_new.GetWeights()
    static_folder_path = "{0}/{1}/static/".format(train_new.train_config['TrainingLogDir'], train_new.train_config['ModelConfig']['ModelName'])

    skel_loss_all = []
    idx_loss_all = []
    skel_accuracy_all = []
    idx_accuracy_all = [] 
    best_w = 0
    best_w_skl = 0

    for step in tqdm(range(1, train_new.train_config["MaxSteps"])): 
        batch_idx_loss, batch_skel_loss = None, None
        
        for batch_step in range(train_new.train_config["BatchSize"]):
            s = time.time()
            idx_loss, skel_loss = train_new.train_step()
            #print("\n 0", time.time() - s)
            train_new.train_counter += 1
            
            if batch_step == 0:
                batch_idx_loss = idx_loss
                batch_skel_loss = skel_loss
            else:
                batch_idx_loss += idx_loss
                batch_skel_loss += skel_loss


        #if train_new.train_config["ModelConfig"]["UseSkel"] and step % 2 == 0:
        #        train_new.backward_step(batch_skel_loss)
        #if train_new.train_config["ModelConfig"]["UseIdx"] and step % 2 == 0:
        #        train_new.backward_step(batch_idx_loss)
        if train_new.train_config["ModelConfig"]["UseSkel"]:
            train_new.backward_step(batch_skel_loss)
            print(batch_skel_loss)
        if train_new.train_config["ModelConfig"]["UseIdx"]:
            #train_new.model.float()
            train_new.backward_step(batch_idx_loss)
            print(batch_idx_loss)

            
        if step % train_new.train_config["LogAfter"] == 0:
            if train_new.train_config["use_cuda"]:
                if train_new.train_config["ModelConfig"]["UseIdx"]:
                    idx_loss_all.append(float(batch_idx_loss.detach().cpu().numpy()))
                
                if train_new.train_config["ModelConfig"]["UseSkel"]:
                    skel_loss_all.append(float(batch_skel_loss.detach().cpu().numpy()))
                
                if train_new.train_config["ModelConfig"]["UseSeg"]:
                    seg_loss_all.append(float(batch_seg_loss.detach().cpu().numpy()))
            else:
                if train_new.train_config["ModelConfig"]["UseIdx"]:
                    idx_loss_all.append(float(batch_idx_loss.detach().numpy()))
                
                if train_new.train_config["ModelConfig"]["UseSkel"]:
                    skel_loss_all.append(float(batch_skel_loss.detach().numpy()))
                
                if train_new.train_config["ModelConfig"]["UseSeg"]:
                    seg_loss_all.append(float(batch_seg_loss.detach().numpy()))
            
            
            save_path = "{0}/{1}/{2}".format(train_new.train_config["TrainingLogDir"], 
                         train_new.train_config["ModelConfig"]["ModelName"], 
                         train_new.train_config["modelsRootName"])
             
             
            saved_path = TrainUtils.save_model(train_new.model, save_path, train_new.train_counter)
             
            ts = test_summary(train_new.test_dentlyset, saved_path, train_new.train_config["TrainingLogDir"],\
                         train_new.train_config["ModelConfig"]["ModelName"], "model.py", train_new.train_counter)
             
            ts.start(train_new.train_config["TestSteps"], use_cuda = train_new.train_config["use_cuda"])
            ts.write_sammary()
            if train_new.train_config["ModelConfig"]["UseIdx"]:
                idx_accuracy_all.append(ts.accuracyIndexAll)
                if idx_accuracy_all[-1] > best_w:
                    best_w = np.around(idx_accuracy_all[-1], decimals=3)
                    best_n = train_new.train_counter
                TrainUtils.save_data_as_graph(idx_loss_all, "{0}/idx_loss_all.png".format(static_folder_path), 'Indexing Loss')
                TrainUtils.save_data_as_graph(idx_accuracy_all, "{0}/idx_accuracy_all.png".format(static_folder_path), 'Best Indexing Accuracy ' + np.str(best_w) +' W number ' + np.str(best_n))
            if train_new.train_config["ModelConfig"]["UseSkel"]:
                skel_accuracy_all.append(ts.accuracySkelAll)
                if skel_accuracy_all[-1] > best_w_skl:
                   best_w_skl = np.around(skel_accuracy_all[-1], decimals=3)
                   best_n = train_new.train_counter
                TrainUtils.save_data_as_graph(skel_loss_all, "{0}/skel_loss_all.png".format(static_folder_path), 'Skeleton Loss')
                TrainUtils.save_data_as_graph(skel_accuracy_all, "{0}/skel_accuracy_all.png".format(static_folder_path), 'Best Skeleton Accuracy' + np.str(best_w_skl)+ 'W number ' + np.str(best_n))
            #TrainUtils.save_data_as_graph(seg_loss_all, "{0}/seg_loss_all.png".format(static_folder_path), 'Segmentation Loss')
    

if __name__ == '__main__':
    mainTrainLoop()

         
         





