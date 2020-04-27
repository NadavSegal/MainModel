import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
import numpy as np
import cv2

class DentlyNet(nn.Module):
    def __init__(self, input_x=0, output_channels=14, is_training=True, batch_size=1):
        super(DentlyNet, self).__init__()
        self.input_x = input_x
        self.batch_size=batch_size
        self.output_channels = output_channels
        self.ks = 3
        self.device = torch.cuda.is_available()
        ### indexing model:
        # P = ((S-1)*W-S+F)/2, with F = filter size, S = stride
        self.down0_conv = nn.Conv2d(3, 9, kernel_size=(37, 9), stride=(1, 1), padding=(18, 4)) #320  3X31X9X9 = 7533
        self.down0_conv_bn = nn.BatchNorm2d(9, track_running_stats=False, momentum=0.5)

        self.mp2d = torch.nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0), return_indices=True)

        self.down1_conv = nn.Conv2d(9, 24, kernel_size=(7, 5), stride=(1, 1), padding=(3, 2)) #160 9X7X5X24 = 7560
        self.down1_conv_bn = nn.BatchNorm2d(24, track_running_stats=False, momentum=0.5)

        self.down2_conv = nn.Conv2d(24, 24, kernel_size=(9, 5), stride=(1, 1), padding=(4, 2))  #24X31X9X24 = 160704
        self.down2_conv_bn = nn.BatchNorm2d(24, track_running_stats=False, momentum=0.5)

        self.down2_1_conv = nn.Conv2d(24, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) #24X3X3X96 = 20736
        self.down2_1_conv_bn = nn.BatchNorm2d(48, track_running_stats=False, momentum=0.5)

        self.down2_2_conv = nn.Conv2d(48, 24, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)) #96X1X1X24 = 2304
        self.down2_2_conv_bn = nn.BatchNorm2d(24, track_running_stats=False, momentum=0.5)

        self.down4_conv = nn.Conv2d(24, 9, kernel_size=(7, 5), stride=(1, 1), padding=(3, 2)) #24X7X5X9 = 7560
        self.down4_conv_bn = nn.BatchNorm2d(9, track_running_stats=False, momentum=0.5)

        self.Unmp2d = torch.nn.MaxUnpool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))

        self.down3_conv = nn.Conv2d(9, output_channels, kernel_size=(5, 3), stride=(1, 1), padding=(2, 1)) #320 9*5*3*14 = 1890
        self.down3_conv_bn = nn.BatchNorm2d(output_channels, track_running_stats=False, momentum=0.5)

        self.down5_conv = nn.Conv2d(output_channels, output_channels, kernel_size=(5, 3), stride=(1, 1), padding=(2, 1)) # 14*5*3*14 = 2940
        self.down5_conv_bn = nn.BatchNorm2d(output_channels, track_running_stats=False, momentum=0.5)

        self.upSample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)


        ### skeleton model:
        # P = ((S-1)*W-S+F)/2, with F = filter size, S = stride
        self.down0_convSkel = nn.Conv2d(3, 7, kernel_size=(31, 5), stride=(1, 1), padding=(15, 2))
        self.down0_conv_bnSkel = nn.BatchNorm2d(7, track_running_stats=False, momentum=0.5)

        self.down1_convSkel = nn.Conv2d(7, 7, kernel_size=(31, 7), stride=(1, 1), padding=(15, 3))
        self.down1_conv_bnSkel = nn.BatchNorm2d(7, track_running_stats=False, momentum=0.5)

        self.down1_convSkel = nn.Conv2d(7, 7, kernel_size=(31, 7), stride=(1, 1), padding=(15, 3))
        self.down1_conv_bnSkel = nn.BatchNorm2d(7, track_running_stats=False, momentum=0.5)

        self.down2_convSkel = nn.Conv2d(7, 1, kernel_size=(31, 7), stride=(1, 1), padding=(15, 3))

        self.mp2dSkel = torch.nn.MaxPool2d(kernel_size=(40, 1), stride=(40, 1), padding=(0, 0), return_indices=True)
        self.Unmp2dSkel = torch.nn.MaxUnpool2d(kernel_size=(40, 1), stride=(40, 1), padding=(0, 0))

        self.mp2dSkel2 = torch.nn.MaxPool2d(kernel_size=(40, 1), stride=(40, 1), padding=(20, 0), return_indices=True)
        self.Unmp2dSkel2 = torch.nn.MaxUnpool2d(kernel_size=(40, 1), stride=(40, 1), padding=(20, 0))


        if self.device:
            self.dtype = torch.cuda.FloatTensor
            self.dtype_l = torch.cuda.LongTensor
        else:
            self.dtype = torch.FloatTensor
            self.dtype_l = torch.LongTensor


    def forward(self, x=torch.rand(1, 3, 640, 640), requires_grad=True):
        if self.device:
            x = Variable(x, requires_grad=requires_grad).cuda()
        else:
            x = Variable(x, requires_grad=requires_grad)

        ### indexing model:
        x_ind = F.interpolate(x, size=160)

        x_ind = self.down0_conv(x_ind)
        x_ind = self.down0_conv_bn(x_ind)
        x_ind = F.leaky_relu(x_ind)

        x_ind, index = self.mp2d(x_ind)

        x_ind = self.down1_conv(x_ind)
        x_ind = self.down1_conv_bn(x_ind)
        x_ind = F.leaky_relu(x_ind)

        x_ind = self.down2_conv(x_ind)
        x_ind = self.down2_conv_bn(x_ind)
        x_ind = F.leaky_relu(x_ind)

        x_ind = self.down2_1_conv(x_ind)
        x_ind = self.down2_1_conv_bn(x_ind)
        x_ind = F.leaky_relu(x_ind)

        x_ind = self.down2_2_conv(x_ind)
        x_ind = self.down2_2_conv_bn(x_ind)
        x_ind = F.leaky_relu(x_ind)

        x_ind = self.down4_conv(x_ind)
        x_ind = self.down4_conv_bn(x_ind)
        x_ind = F.leaky_relu(x_ind)

        x_ind = self.Unmp2d(x_ind, index)

        x_ind = self.down3_conv(x_ind)
        x_ind = self.down3_conv_bn(x_ind)
        x_ind = F.leaky_relu(x_ind)

        x_ind = self.down5_conv(x_ind)
        x_ind = self.down5_conv_bn(x_ind)
        x_ind = F.leaky_relu(x_ind)

        x_ind = self.upSample(x_ind)
        x_ind = F.leaky_relu(x_ind)

        indexing_tensor = x_ind.narrow(1, 0, self.output_channels)  # [:,:49,:,:]
        indexing_tensor = torch.exp(F.log_softmax(indexing_tensor, 1))
        indexing_pred, indexing_argmax = indexing_tensor.max(dim=1)

        ## skeleton model:

        x1 = self.down0_convSkel(x)
        x2 = self.down0_conv_bnSkel(x1)
        x3 = F.relu(self.down1_convSkel(x2))
        x4 = self.down1_conv_bnSkel(x3)
        x5 = F.relu(x4)
        x6 = self.down2_convSkel(x5)
        x7 = F.relu(x6)

        x8_1, index = self.mp2dSkel(x7)
        x8_2 = self.Unmp2dSkel(x8_1, index)
        x9_1, index = self.mp2dSkel2(x7)
        x9_2 = self.Unmp2dSkel2(x9_1, index)
        x8_2[0, 0, 0:10, :] = x9_2[0, 0, 0:10, :]
        x8_2[0, 0, 30:50, :] = x9_2[0, 0, 30:50, :]
        x8_2[0, 0, 70:90, :] = x9_2[0, 0, 70:90, :]
        x8_2[0, 0, 110:130, :] = x9_2[0, 0, 110:130, :]
        x8_2[0, 0, 150:170, :] = x9_2[0, 0, 150:170, :]
        x8_2[0, 0, 190:210, :] = x9_2[0, 0, 190:210, :]
        x8_2[0, 0, 230:250, :] = x9_2[0, 0, 230:250, :]
        x8_2[0, 0, 270:290, :] = x9_2[0, 0, 270:290, :]
        x8_2[0, 0, 270:290, :] = x9_2[0, 0, 270:290, :]
        x8_2[0, 0, 310:330, :] = x9_2[0, 0, 310:330, :]
        x8_2[0, 0, 350:370, :] = x9_2[0, 0, 350:370, :]
        x8_2[0, 0, 390:410, :] = x9_2[0, 0, 390:410, :]
        x8_2[0, 0, 430:450, :] = x9_2[0, 0, 430:450, :]
        x8_2[0, 0, 470:490, :] = x9_2[0, 0, 470:490, :]
        x8_2[0, 0, 510:530, :] = x9_2[0, 0, 510:530, :]
        x8_2[0, 0, 550:570, :] = x9_2[0, 0, 550:570, :]
        x8_2[0, 0, 590:610, :] = x9_2[0, 0, 590:610, :]
        x8_2[0, 0, 630:640, :] = x9_2[0, 0, 630:640, :]

        skeleton_pred = x7
        skeleton_final = x8_2
        final_res = torch.clone(indexing_argmax)
        final_res[0, skeleton_final[0, 0, :, :] == 0] = 0

        return final_res, skeleton_final, skeleton_pred, indexing_tensor, indexing_argmax, indexing_pred


    def forward_eval(self, x=torch.rand(4, 3, 640, 640).cuda(), requires_grad=True):
    #        x = torch.from_numpy(x)
        with torch.no_grad(): 
            x_00 = self.down0_conv_bn(self.down0_conv(x))
            x_10 = self.down1_conv_bn(self.down1_conv(x_00))
            x = self.down2_conv_bn(self.down2_conv(x_10))
            x = self.down3_conv_bn(self.down3_conv(x))
            x_20 = self.down4_conv_bn(self.down4_conv(x))
            x = self.down5_conv_bn(self.down5_conv(x_20))
            x = self.down6_conv_bn(self.down6_conv(x))
            x_30 = self.down7_conv_bn(self.down7_conv(x))
            x = self.down8_conv_bn(self.down8_conv(x_30))
            x = self.down9_conv_bn(self.down9_conv(x))
            x = self.down10_conv_bn(self.down10_conv(x))
            x = self.upsample1_ConvTranspose2d(x, output_size=(40, 40))
            x_31 = self.upsample2_conv_bn(self.upsample2_conv(x))
            x = torch.cat((x_30, x_31), 1)
            x = self.upsample3_conv_bn(self.upsample3_conv(x))  
            x = self.upsample4_conv_bn(self.upsample4_conv(x))
            
            
            
            x = self.upsample5_ConvTranspose2d(x, output_size=(80, 80)) 
            x_21 = self.upsample7_conv_bn(self.upsample7_conv(x))
            x = torch.cat((x_20, x_21), 1)
            x = self.upsample8_conv_bn(self.upsample8_conv(x))  
            x = self.upsample9_conv_bn(self.upsample9_conv(x) ) 
            x = self.upsample10_ConvTranspose2d(x, output_size=(160, 160)) 
            x_11 = self.upsample11_conv_bn(self.upsample11_conv(x))  
            x = torch.cat((x_10, x_11), 1)
            x = self.upsample12_conv_bn(self.upsample12_conv(x)) 
            x = self.upsample13_conv_bn(self.upsample13_conv(x)) 
            x = self.upsample14_ConvTranspose2d(x, output_size=(320, 320)) 
            x_01 = self.upsample15_conv_bn(self.upsample15_conv(x)) 
            x = torch.cat((x_00, x_01), 1)
            x = F.relu(self.upsample16_conv_bn(self.upsample16_conv(x)))
            x = F.relu(self.upsample17_conv_bn(self.upsample17_conv(x)))
            x = F.relu(self.upsample18_ConvTranspose2d(x, output_size=(640, 640)))
            x = F.relu(self.upsample19_conv(x))
            x = self.upsample20_conv(x)
            
            indexing_pred= x.narrow(1, 0, 49)#[:,:49,:,:]
            seg_pred = x.narrow(1, 49, 3)
            
            indexing_pred, seg_pred = torch.exp(F.log_softmax(indexing_pred, 1)), torch.exp(F.log_softmax(seg_pred, 1))
            indexing_output, indexing_argmax = indexing_pred.max(dim=1)
            seg_output, seg_argmax = seg_pred.max(dim=1)
    #        print(self.output.shape
            return indexing_output, indexing_argmax, seg_output, seg_argmax, None
        
        
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
        
        skeleton_aslist = np.concatenate((idxs_uint8, values), 1)
        
        if idxs_uint8.shape[0] < 15000:
            size = 15000 - idxs_uint8.shape[0]
            pad_with_zeros = np.zeros((size, 5),'uint8')       
            skeleton_aslist = np.concatenate((skeleton_aslist, pad_with_zeros), 0)
        else:
            skeleton_aslist = skeleton_aslist[0:15000-1,:];
        
            
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
#a.eval()
#a.load_Weights('/home/alon/dently_7_19/traind_model_129000.pt')
##
#import time 
#start = time.time()
#indexing_output, indexing_argmax, seg_output, seg_argmax = a.forward_eval()
#print(time.time() - start)