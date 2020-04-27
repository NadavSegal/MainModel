import torch
import torch.nn as nn
import torch.nn.functional as F
from losses import dont_care_crossentropy

from torch.autograd import Variable
import numpy as np
import cv2

class DentlyNet(nn.Module):
    def __init__(self, input_x=0, output_channels=49+3, is_training=True, batch_size=1):
        super(DentlyNet, self).__init__()
        self.input_x = input_x
        self.batch_size=batch_size
        self.output_channels = output_channels
        self.ks = 3
        self.device = torch.cuda.is_available()
        

        self.down0_conv = nn.Conv2d(3, 12, kernel_size=(13,13), bias=True, stride=(1, 1), padding=6)
        self.bn0 = nn.BatchNorm2d(12, track_running_stats=False, momentum=0.5)
        self.down1_conv = nn.Conv2d(12, 24, kernel_size=(3,3), stride=(1, 1), padding=1)
        self.bn1 = nn.BatchNorm2d(24, track_running_stats=False, momentum=0.5)
        self.down2_conv = nn.Conv2d(36, 36, kernel_size=(5,5), stride=(1, 1), padding=2)
        self.bn2 = nn.BatchNorm2d(36, track_running_stats=False, momentum=0.5)
        self.down3_conv = nn.Conv2d(36, output_channels, kernel_size=(1,1), stride=(1, 1), padding=0)
        self.bn3 = nn.BatchNorm2d(output_channels, track_running_stats=False, momentum=0.5)
        self.down4_conv = nn.Conv2d(output_channels, output_channels, kernel_size=(3,3), stride=(1, 1), padding=1)
        self.bn4 = nn.BatchNorm2d(output_channels, track_running_stats=False, momentum=0.5)
        self.down5_conv = nn.Conv2d(output_channels, output_channels, kernel_size=(5,5), stride=(1, 1), padding=2)
        self.bn5 = nn.BatchNorm2d(output_channels, track_running_stats=False, momentum=0.5)
        self.down6_conv = nn.Conv2d(104, output_channels, kernel_size=(1,1), stride=(1, 1), padding=0)
        self.bn6 = nn.BatchNorm2d(output_channels, track_running_stats=False, momentum=0.5)
        
        if self.device:
            self.dtype = torch.cuda.FloatTensor
            self.dtype_l = torch.cuda.LongTensor
        else:
            self.dtype = torch.FloatTensor
            self.dtype_l = torch.LongTensor
            
        self.loss = dont_care_crossentropy()

    def forward(self, x=torch.rand(1, 3, 640, 640), requires_grad=True):
#        x_111 = torch.from_numpy(x)
#        
#        x = Variable(x_111, requires_grad=requires_grad).cuda()
#        
        x = torch.from_numpy(x)
        
        x = Variable(x, requires_grad=requires_grad).cuda()
        x = F.leaky_relu(self.bn0(self.down0_conv(x)))
        x1 = F.leaky_relu(self.bn1(self.down1_conv(x)))
        x = torch.cat((x, x1), 1)
        x = F.leaky_relu(self.bn2(self.down2_conv(x)))
        x = F.leaky_relu(self.bn3(self.down3_conv(x)))
        x1 = F.leaky_relu(self.bn4(self.down4_conv(x)))
        x = F.leaky_relu(self.bn5(self.down5_conv(x)))
        x = torch.cat((x, x1), 1)
        x = F.leaky_relu(self.down6_conv(x))
        
        
        indexing_pred= x.narrow(1, 0, 49)#[:,:49,:,:]
        seg_pred = x.narrow(1, 49, 3)
        
        return indexing_pred, seg_pred, []
    
    def forward_eval(self, x=torch.rand(4, 3, 640, 640), requires_grad=True):
    #        x = torch.from_numpy(x)
        with torch.no_grad(): 
            x = Variable(x, requires_grad=requires_grad).cuda()
            x = F.leaky_relu(self.bn0(self.down0_conv(x)))
            x1 = F.leaky_relu(self.bn1(self.down1_conv(x)))
            x = torch.cat((x, x1), 1)
            x = F.leaky_relu(self.bn2(self.down2_conv(x)))
            x = F.leaky_relu(self.bn3(self.down3_conv(x)))
            x1 = F.leaky_relu(self.bn4(self.down4_conv(x)))
            x = F.leaky_relu(self.bn5(self.down5_conv(x)))
            x = torch.cat((x, x1), 1)
            x = F.leaky_relu(self.down6_conv(x))
            
            indexing_pred= x.narrow(1, 0, 49)#[:,:49,:,:]
            seg_pred = x.narrow(1, 49, 3)
            
            indexing_pred, seg_pred = torch.exp(F.log_softmax(indexing_pred, 1)), torch.exp(F.log_softmax(seg_pred, 1))
            indexing_output, indexing_argmax = indexing_pred.max(dim=1)
            seg_output, seg_argmax = seg_pred.max(dim=1)
    #        print(self.output.shape
            return indexing_output, indexing_argmax, seg_output, seg_argmax, []
        
        
class Eval:
    def __init__(self):        
        self.model  = DentlyNet()
        self.model.eval()
        self.mp2d = torch.nn.MaxPool2d(kernel_size = (7,1), stride=(1, 1), padding=(3,0))
        self.gpu=torch.cuda.is_available()
        if self.gpu:
            self.dtype = torch.cuda.FloatTensor
            self.dtype_l = torch.cuda.LongTensor
        else:
            self.dtype = torch.FloatTensor
            self.dtype_l = torch.LongTensor
        
    def igal_try(self,
                 src_image):    
        
        return 2*src_image    
    
    def load_Weights(self, pt_trained_model_path=None):
        '''
        pt_trained_model_path: path to trainrd pt file for example: ./traind_model_316000.pt'
        '''
        
        if pt_trained_model_path:
            print("loading from trained model path: {0}".format(pt_trained_model_path))
            
            if self.gpu:
                
                model_dict = self.model.state_dict()
                self.model.load_state_dict(model_dict)
                pretrained_dict = torch.load(pt_trained_model_path) 

                # 1. filter out unnecessary keys
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}# and not "conv_59_32_32_output" in k.split(".") and not "ce3" in k.split(".")}
                # 2. overwrite entries in the existing state dict
                model_dict.update(pretrained_dict) 

                self.model.load_state_dict(model_dict)
#                self.model.load_state_dict(torch.load(pt_trained_model_path))
                try:
                    self.model.cuda()
                except:
                    print("Can't load the model on gpu")
                    print("using cpu mode")
            else:
                self.model.load_state_dict(torch.load(pt_trained_model_path, map_location={'cuda:0': 'cpu'}))
        else:
            print("Please check if the model get the correct pt file path")
  
    
    def img_to_input_format(self, x=None):
        x = torch.from_numpy(x).type(self.dtype)
        x = x.permute(2,0,1)
        x = torch.div(x , 255.0)
        x = torch.unsqueeze(x, 0)
            
        return x
    
    def from_img(self, real_image):
#        for i in range(3):
#            real_image = dentlyset.load_mat_file_as_np(x_path, mode=0)
#            x, _ = img_and_labels_to_model_input_format(real_image)
#            x_pred = self.model.forward_eval(x)
#        
            
#        real_image = dentlyset.load_mat_file_as_np(x_path, mode=0)
#        real_image = cv2.cvtColor(real_image, cv2.COLOR_BGR2RGB)
        temp1 = np.zeros((1280,1280, 3))
        s1 = np.random.randint(0,630)
        s2 = np.random.randint(0,630)
        
        temp1[s1:s1 + 640, s2:s2+640, :] = real_image.copy()
        real_image = temp1
        
        x = self.img_to_input_format(real_image)
        indexing_output, indexing_argmax, seg_output, seg_argmax = self.model.forward_eval(x)
        indexing_output, indexing_argmax = self._process_indexing(indexing_output, indexing_argmax)
        
        skeleton, = self._process_skelaton(indexing_output, indexing_argmax)
        with torch.no_grad():
            torch.cuda.synchronize()
            skeleton_o = skeleton.cpu().data.numpy()
            
        values = skeleton_o[skeleton_o.nonzero()]
        values = values[..., None]
        idxs = np.argwhere(skeleton_o > 0)
        
        idxs_uint8 = np.zeros((np.shape(idxs)[0],4),'uint8')
        idxs_uint8[:,0] = idxs[:,0] / 256
        idxs_uint8[:,1] = idxs[:,0] % 256
        idxs_uint8[:,2] = idxs[:,1] / 256
        idxs_uint8[:,3] = idxs[:,1] % 256       
        
        size = 15000 -idxs_uint8.shape[0]
        pad_with_zeros = np.zeros((size, 5),'uint8')
        skeleton_aslist = np.concatenate((idxs_uint8, values), 1)
        skeleton_aslist = np.concatenate((skeleton_aslist, pad_with_zeros), 0)
        indexing_output = indexing_output.cpu().detach().numpy()  
        indexing_argmax = indexing_argmax.cpu().detach().numpy()
        
        return real_image, skeleton_o, indexing_output, indexing_argmax, seg_output, seg_argmax, skeleton_aslist.astype('uint8')
        
    
    def _process_indexing(self,
                          indexing_output, 
                          indexing_argmax,
                          th=0.5 ):
        
        value_max = indexing_output
        value_max[value_max < th] = 0
        indexing_output[value_max == 0] = 0
        indexing_argmax[value_max == 0] = 0
#        indexing_argmax[indexing_argmax % 2 == 1] = 0
        
        return indexing_output, indexing_argmax
    
    def _process_seg(self,
                          seg_output, 
                          seg_argmax,
                          th=0.6):
        
        value_max = seg_output.max(axis=-1)
        value_max[value_max < th] = 0
        seg_argmax[value_max == 0] = 0
        seg_argmax[seg_argmax == 2] = 0
        
        return seg_argmax
    
    
    def _process_skelaton(self, process_indexing, indexing_argmax):
        

        test11 = self.mp2d(process_indexing.type(torch.cuda.FloatTensor))
        process_indexing = torch.where((indexing_argmax != 0), process_indexing.type(torch.cuda.FloatTensor), torch.FloatTensor(process_indexing.size(0),  process_indexing.size(1), process_indexing.size(2)).zero_().type(torch.cuda.FloatTensor))

        indexing_output = torch.where((process_indexing == test11), indexing_argmax.type(torch.cuda.FloatTensor), torch.FloatTensor(process_indexing.size(0),  process_indexing.size(1), process_indexing.size(2)).zero_().type(torch.cuda.FloatTensor))

        return indexing_output
    
    
    def _draw_results(self,
                      src_image,
                      mask):
        
        b = mask[..., None] > 0
        bbb = np.concatenate([b, b, b], axis=2)
        
        src_image[bbb > 0] = 0
        
        return src_image
    
#    def load_Weights(self, pt_trained_model_path=None):        
#        pretrained_dict = torch.load(pt_trained_model_path)
#        model_dict = self.state_dict()
#
#        # 1. filter out unnecessary keys
#        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
#        # 2. overwrite entries in the existing state dict
#        model_dict.update(pretrained_dict) 
#        # 3. load the new state dict
##        model.load_state_dict(pretrained_dict)
#        if pt_trained_model_path:
#                print("loading from trained model path: {0}".format(pt_trained_model_path))
#                
#                if self.device:
#                    self.load_state_dict(pretrained_dict)
#                    try:
#                        self.cuda()
#                    except:
#                        print("Can't load the model on gpu")
#                        print("using cpu mode")
#                else:
#                    self.load_state_dict(pretrained_dict)
#        else:
#            print("Please check if the model get the correct pt file path")
#
#a=DentlyNet()
#a()
#a.load_Weights('/home/alon/dently_7_19/traind_model_129000.pt')
##
#import time 
#start = time.time()
#indexing_output, indexing_argmax, seg_output, seg_argmax = a.forward_eval()
#print(time.time() - start)