import torch
import torch.nn as nn
import torch.nn.functional as F
from losses import dont_care_crossentropy

from torch.autograd import Variable
import numpy as np
import cv2

class Skeleton:
    def __init__(self, 
                 label, 
                 sub_label=None,
                 max_i=48,
                 min_i=2):
        
        indices = (label).nonzero().squeeze()
        
        self.y = indices[:, 0].type(torch.LongTensor)
        self.x = indices[:, 1].type(torch.LongTensor)
        self.idxs = label[self.y, self.x].type(torch.IntTensor)
        
        self.idxs[self.idxs > max_i] = 0
        self.idxs[self.idxs <= min_i] = 0
        
        self.s_list = np.zeros((1, 64, 640))
        self.s_list[:,self.idxs, self.x] += self.y
        
class DentlyNet(nn.Module):
    def __init__(self, input_x=0, output_channels=49+3, is_training=True, batch_size=1):
        super(DentlyNet, self).__init__()
        self.input_x = input_x
        self.batch_size=batch_size
        self.output_channels = output_channels
        self.ks = 3
        self.device = torch.cuda.is_available()
        
        self.avg_pool =  nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.down1_conv = nn.Conv2d(3, 3, kernel_size=(3,3), stride=(2, 2), padding=1)
        self.down1_conv_bn = nn.BatchNorm2d(3, track_running_stats=False, momentum=0.5)
#        self.down00_conv = nn.Conv2d(3, 3, kernel_size=(5,5), stride=(2, 2), padding=2)
#        self.down00_conv_bn = nn.BatchNorm2d(3, track_running_stats=False, momentum=0.5)
#        self.down000_conv = nn.Conv2d(3, 3, kernel_size=(1,1), stride=(2, 2))
#        self.down000_conv_bn = nn.BatchNorm2d(3, track_running_stats=False, momentum=0.5)
        
        
        self.down1_conv = nn.Conv2d(3, 6, kernel_size=(3,3), stride=(2, 2), padding=1)
        self.down1_conv_bn = nn.BatchNorm2d(6, track_running_stats=False, momentum=0.5)
        self.down2_conv = nn.Conv2d(6, 16, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.down2_conv_bn = nn.BatchNorm2d(16, track_running_stats=False, momentum=0.5)
        self.down3_conv = nn.Conv2d(16, 16, kernel_size=(self.ks, self.ks), stride=(1, 1), padding=1)
        self.down3_conv_bn = nn.BatchNorm2d(16, track_running_stats=False, momentum=0.5)
        self.down4_conv = nn.Conv2d(16, 16, kernel_size=(1,1), stride=(2, 2))
        self.down4_conv_bn = nn.BatchNorm2d(16, track_running_stats=False, momentum=0.5)
        self.down5_conv = nn.Conv2d(16, 32, kernel_size=(self.ks, self.ks), stride=(1, 1), padding=1)
        self.down5_conv_bn = nn.BatchNorm2d(32, track_running_stats=False, momentum=0.5)
        self.down6_conv = nn.Conv2d(32, 32, kernel_size=(self.ks, self.ks), stride=(1, 1), padding=1)
        self.down6_conv_bn = nn.BatchNorm2d(32, track_running_stats=False, momentum=0.5)
        self.down7_conv = nn.Conv2d(32, 64, kernel_size=(1,1), stride=(2, 2))
        self.down7_conv_bn = nn.BatchNorm2d(64, track_running_stats=False, momentum=0.5)
        self.down8_conv = nn.Conv2d(64, 64, kernel_size=(self.ks, self.ks), stride=(1, 1), padding=1)
        self.down8_conv_bn = nn.BatchNorm2d(64, track_running_stats=False, momentum=0.5)
        self.down9_conv = nn.Conv2d(64, 64, kernel_size=(self.ks, self.ks), stride=(1, 1), padding=1)
        self.down9_conv_bn = nn.BatchNorm2d(64, track_running_stats=False, momentum=0.5)
        self.down10_conv = nn.Conv2d(64, 64, kernel_size=(1,1), stride=(2, 2))
        self.down10_conv_bn = nn.BatchNorm2d(64, track_running_stats=False, momentum=0.5)
        self.upsample1_ConvTranspose2d = nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1)
        self.upsample2_conv  = nn.Conv2d(64, 128, kernel_size=(1,1), stride=(1, 1))
        self.upsample2_conv_bn = nn.BatchNorm2d(128, track_running_stats=False, momentum=0.5)
        self.upsample3_conv  = nn.Conv2d(192, 128, kernel_size=(self.ks, self.ks), stride=(1, 1), padding=1)
        self.upsample3_conv_bn = nn.BatchNorm2d(128, track_running_stats=False, momentum=0.5)
        self.upsample4_conv  = nn.Conv2d(128, 128, kernel_size=(1,1), stride=(1, 1))
        self.upsample4_conv_bn = nn.BatchNorm2d(128, track_running_stats=False, momentum=0.5)
        self.upsample5_ConvTranspose2d = nn.ConvTranspose2d(128, 128, 3, stride=2, padding=1)
        self.upsample7_conv  = nn.Conv2d(128, 64, kernel_size=(1,1), stride=(1, 1))
        self.upsample7_conv_bn = nn.BatchNorm2d(64, track_running_stats=False, momentum=0.5)
        self.upsample8_conv  = nn.Conv2d(80, 64, kernel_size=(self.ks, self.ks), stride=(1, 1), padding=1)
        self.upsample8_conv_bn = nn.BatchNorm2d(64, track_running_stats=False, momentum=0.5)
        self.upsample9_conv  = nn.Conv2d(64, 64, kernel_size=(1,1), stride=(1, 1))
        self.upsample9_conv_bn = nn.BatchNorm2d(64, track_running_stats=False, momentum=0.5)
        self.upsample10_ConvTranspose2d = nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1)
        self.upsample11_conv  = nn.Conv2d(64, 64, kernel_size=(1,1), stride=(1, 1))
        self.upsample11_conv_bn = nn.BatchNorm2d(64, track_running_stats=False)
        self.upsample12_conv  = nn.Conv2d(70, 64, kernel_size=(self.ks, self.ks), stride=(1, 1), padding=1)
        self.upsample12_conv_bn = nn.BatchNorm2d(64, track_running_stats=False)
        self.upsample13_conv  = nn.Conv2d(64, 32, kernel_size=(1,1), stride=(1, 1))
        self.upsample13_conv_bn = nn.BatchNorm2d(32, track_running_stats=False, momentum=0.5)
        self.upsample14_ConvTranspose2d = nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1)
        self.upsample15_conv  = nn.Conv2d(32, 16, kernel_size=(1,1), stride=(1, 1))
        self.upsample15_conv_bn = nn.BatchNorm2d(16, track_running_stats=False, momentum=0.5)
        self.upsample16_conv  = nn.Conv2d(19, 32, kernel_size=(self.ks, self.ks), stride=(1, 1), padding=1)
        self.upsample16_conv_bn = nn.BatchNorm2d(32, track_running_stats=False, momentum=0.5)
        self.upsample17_conv  = nn.Conv2d(32, 32, kernel_size=(1,1), stride=(1, 1))
        self.upsample17_conv_bn = nn.BatchNorm2d(32, track_running_stats=False, momentum=0.5)
        self.upsample18_ConvTranspose2d = nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1)
        self.upsample19_conv  = nn.Conv2d(32, 16, kernel_size=(1,1), stride=(1, 1))
        self.upsample20_conv  = nn.Conv2d(16, self.output_channels, kernel_size=(self.ks, self.ks), stride=(1, 1), padding=1)
        if self.device:
            self.dtype = torch.cuda.FloatTensor
            self.dtype_l = torch.cuda.LongTensor
        else:
            self.dtype = torch.FloatTensor
            self.dtype_l = torch.LongTensor
            
        self.loss = dont_care_crossentropy()

    def forward(self, x=np.random.random((1, 3, 1280, 800)), requires_grad=True):
#        x_111 = torch.from_numpy(x)
            t = 1
#        x = Variable(x_111, requires_grad=requires_grad).cuda()
                        

            if self.device:
                x = torch.from_numpy(x).type(torch.cuda.FloatTensor)
            
            x = Variable(x, requires_grad=requires_grad) #.cuda()
            if self.device:
                x = x.cuda()

            

#            x_00 = self.down0_conv_bn(self.down0_conv(x))
            x_00 = self.avg_pool(x)
            x_10 = self.down1_conv_bn(self.down1_conv(x_00))
            
    #        x_000 = self.down00_conv_bn(self.down00_conv(x))
#            x_0000 = self.down000_conv_bn(self.down000_conv(x))
#            x_00 = torch.cat((x_00, x_0000), 1)
        
        
#            x_10 = self.down1_conv_bn(self.down1_conv(x_00))
            x = self.down2_conv_bn(self.down2_conv(x_10))
            x = self.down3_conv_bn(self.down3_conv(x))
            x_20 = self.down4_conv_bn(self.down4_conv(x))
            x = self.down5_conv_bn(self.down5_conv(x_20))
            x = self.down6_conv_bn(self.down6_conv(x))
            x_30 = self.down7_conv_bn(self.down7_conv(x))
            x = self.down8_conv_bn(self.down8_conv(x_30))
            x = self.down9_conv_bn(self.down9_conv(x))
            x = self.down10_conv_bn(self.down10_conv(x))
            if t == 1:
                x = self.upsample1_ConvTranspose2d(x, output_size=(80, 50))
            else:
                x = self.upsample1_ConvTranspose2d(x, output_size=(50, 80))
                
            
            x_31 = self.upsample2_conv_bn(self.upsample2_conv(x))
            x = torch.cat((x_30, x_31), 1)
            x = self.upsample3_conv_bn(self.upsample3_conv(x))  
            x = self.upsample4_conv_bn(self.upsample4_conv(x))
            
            
            if t == 1:
                x = self.upsample5_ConvTranspose2d(x, output_size=(160, 100)) 
            else:
                x = self.upsample5_ConvTranspose2d(x, output_size=(100, 160)) 
                
            x_21 = self.upsample7_conv_bn(self.upsample7_conv(x))
            x = torch.cat((x_20, x_21), 1)
            x = self.upsample8_conv_bn(self.upsample8_conv(x))  
            x = self.upsample9_conv_bn(self.upsample9_conv(x) ) 
            if t == 1:
                x = self.upsample10_ConvTranspose2d(x, output_size=(320, 200)) 
            else:
                x = self.upsample10_ConvTranspose2d(x, output_size=(200, 320)) 
                
            x_11 = self.upsample11_conv_bn(self.upsample11_conv(x))  
            x = torch.cat((x_10, x_11), 1)
            x = self.upsample12_conv_bn(self.upsample12_conv(x)) 
            x = self.upsample13_conv_bn(self.upsample13_conv(x)) 
            if t == 1:
                x = self.upsample14_ConvTranspose2d(x, output_size=(640, 400)) 
            else:
                x = self.upsample14_ConvTranspose2d(x, output_size=(400, 640)) 
                
                
            x_01 = self.upsample15_conv_bn(self.upsample15_conv(x)) 
            x = torch.cat((x_00, x_01), 1)
            x = F.relu(self.upsample16_conv_bn(self.upsample16_conv(x)))
            x = F.relu(self.upsample17_conv_bn(self.upsample17_conv(x)))
            if t == 1:
                x = F.relu(self.upsample18_ConvTranspose2d(x, output_size=(1280, 800)))
            else:
                x = F.relu(self.upsample18_ConvTranspose2d(x, output_size=(800, 1200)))
                
            x = F.relu(self.upsample19_conv(x))
            x = self.upsample20_conv(x)
            
            indexing_pred= x.narrow(1, 0, 49)#[:,:49,:,:]
            seg_pred = x.narrow(1, 49, 3)
            
            return indexing_pred, seg_pred, []
    
    def forward_eval(self, x=torch.rand(4, 3, 1280, 1280), requires_grad=True):
    #        x = torch.from_numpy(x)
        t = 1
#        if self.device:
#                x = torch.from_numpy(x).type(torch.cuda.FloatTensor
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.no_grad(): 
#            x_00 = self.down0_conv_bn(self.down0_conv(x))
            x_00 = self.avg_pool(x)
            
#    #        x_000 = self.down00_conv_bn(self.down00_conv(x))
#            x_0000 = self.down000_conv_bn(self.down000_conv(x))
#            x_00 = torch.cat((x_00, x_0000), 1)
        
            x_10 = self.down1_conv_bn(self.down1_conv(x_00))
#            x_10 = self.down1_conv_bn(self.down1_conv(x_00))
            x = self.down2_conv_bn(self.down2_conv(x_10))
            x = self.down3_conv_bn(self.down3_conv(x))
            x_20 = self.down4_conv_bn(self.down4_conv(x))
            x = self.down5_conv_bn(self.down5_conv(x_20))
            x = self.down6_conv_bn(self.down6_conv(x))
            x_30 = self.down7_conv_bn(self.down7_conv(x))
            x = self.down8_conv_bn(self.down8_conv(x_30))
            x = self.down9_conv_bn(self.down9_conv(x))
            x = self.down10_conv_bn(self.down10_conv(x))
            if t == 1:
                x = self.upsample1_ConvTranspose2d(x, output_size=(80, 50))
            else:
                x = self.upsample1_ConvTranspose2d(x, output_size=(50, 80))
                
            
            x_31 = self.upsample2_conv_bn(self.upsample2_conv(x))
            x = torch.cat((x_30, x_31), 1)
            x = self.upsample3_conv_bn(self.upsample3_conv(x))  
            x = self.upsample4_conv_bn(self.upsample4_conv(x))
            
            
            if t == 1:
                x = self.upsample5_ConvTranspose2d(x, output_size=(160, 100)) 
            else:
                x = self.upsample5_ConvTranspose2d(x, output_size=(100, 160)) 
                
            x_21 = self.upsample7_conv_bn(self.upsample7_conv(x))
            x = torch.cat((x_20, x_21), 1)
            x = self.upsample8_conv_bn(self.upsample8_conv(x))  
            x = self.upsample9_conv_bn(self.upsample9_conv(x) ) 
            if t == 1:
                x = self.upsample10_ConvTranspose2d(x, output_size=(320, 200)) 
            else:
                x = self.upsample10_ConvTranspose2d(x, output_size=(200, 320)) 
                
            x_11 = self.upsample11_conv_bn(self.upsample11_conv(x))  
            x = torch.cat((x_10, x_11), 1)
            x = self.upsample12_conv_bn(self.upsample12_conv(x)) 
            x = self.upsample13_conv_bn(self.upsample13_conv(x)) 
            if t == 1:
                x = self.upsample14_ConvTranspose2d(x, output_size=(640, 400)) 
            else:
                x = self.upsample14_ConvTranspose2d(x, output_size=(400, 640)) 
                
                
            x_01 = self.upsample15_conv_bn(self.upsample15_conv(x)) 
            x = torch.cat((x_00, x_01), 1)
            x = F.relu(self.upsample16_conv_bn(self.upsample16_conv(x)))
            x = F.relu(self.upsample17_conv_bn(self.upsample17_conv(x)))
            if t == 1:
                x = F.relu(self.upsample18_ConvTranspose2d(x, output_size=(1280, 800)))
            else:
                x = F.relu(self.upsample18_ConvTranspose2d(x, output_size=(800, 1200)))
                
                
            x = F.relu(self.upsample19_conv(x))
            x = self.upsample20_conv(x)
            
            indexing_pred= x.narrow(1, 0, 49)#[:,:49,:,:]
            seg_pred = x.narrow(1, 49, 3)
            
            indexing_pred, seg_pred = torch.exp(F.log_softmax(indexing_pred, 1)), torch.exp(F.log_softmax(seg_pred, 1))
            indexing_output, indexing_argmax = indexing_pred.max(dim=1)
            seg_output, seg_argmax = seg_pred.max(dim=1)
    #        print(self.output.shape
            return indexing_output, indexing_argmax, seg_output, seg_argmax, []
        

#a = DentlyNet().cuda()
#indexing_pred, seg_pred, tt = a()
        
class Eval:
    def __init__(self):        
        self.model  = DentlyNet()
        self.model.eval()
        self.mp2d = torch.nn.MaxPool2d(kernel_size = (9,1), stride=(1, 1), padding=(3,0))
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