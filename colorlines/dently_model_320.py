import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from losses import dont_care_crossentropy
import numpy as np

class DentlyNet(nn.Module):
    def __init__(self, input_x=0, output_channels=49+3, is_training=True, batch_size=1):
        super(DentlyNet, self).__init__()
        self.input_x = input_x
        self.batch_size=batch_size
        self.output_channels = output_channels
        self.ks = 3
        self.device = torch.cuda.is_available()
        

#        self.down0_conv = nn.Conv2d(3, 16, kernel_size=(1,1), stride=(2, 2))
#        self.down0_conv_bn = nn.BatchNorm2d(16, track_running_stats=False, momentum=0.5)
        self.down1_conv = nn.Conv2d(3, 16, kernel_size=(1,1), stride=(2, 2))
        self.down1_conv_bn = nn.BatchNorm2d(16, track_running_stats=False, momentum=0.5)
        self.down2_conv = nn.Conv2d(16, 16, kernel_size=(self.ks, self.ks), stride=(1, 1), padding=1)
        self.down2_conv_bn = nn.BatchNorm2d(16, track_running_stats=False, momentum=0.5)
        self.down3_conv = nn.Conv2d(16, 32, kernel_size=(self.ks, self.ks), stride=(1, 1), padding=1)
        self.down3_conv_bn = nn.BatchNorm2d(32, track_running_stats=False, momentum=0.5)
        self.down4_conv = nn.Conv2d(32, 64, kernel_size=(1,1), stride=(2, 2))
        self.down4_conv_bn = nn.BatchNorm2d(64, track_running_stats=False, momentum=0.5)
        self.down5_conv = nn.Conv2d(64, 32, kernel_size=(self.ks, self.ks), stride=(1, 1), padding=1)
        self.down5_conv_bn = nn.BatchNorm2d(32, track_running_stats=False, momentum=0.5)
        self.down6_conv = nn.Conv2d(32, 64, kernel_size=(self.ks, self.ks), stride=(1, 1), padding=1)
        self.down6_conv_bn = nn.BatchNorm2d(64, track_running_stats=False, momentum=0.5)
        self.down7_conv = nn.Conv2d(64, 64, kernel_size=(1,1), stride=(2, 2))
        self.down7_conv_bn = nn.BatchNorm2d(64, track_running_stats=False, momentum=0.5)
        self.down8_conv = nn.Conv2d(64, 64, kernel_size=(self.ks, self.ks), stride=(1, 1), padding=1)
        self.down8_conv_bn = nn.BatchNorm2d(64, track_running_stats=False, momentum=0.5)
        self.down9_conv = nn.Conv2d(64, 64, kernel_size=(self.ks, self.ks), stride=(1, 1), padding=1)
        self.down9_conv_bn = nn.BatchNorm2d(64, track_running_stats=False, momentum=0.5)
        self.down10_conv = nn.Conv2d(64, 256, kernel_size=(1,1), stride=(2, 2))
        self.down10_conv_bn = nn.BatchNorm2d(256, track_running_stats=False, momentum=0.5)
        self.upsample1_ConvTranspose2d = nn.ConvTranspose2d(256, 256, 3, stride=2, padding=1)
        self.upsample2_conv  = nn.Conv2d(256, 128, kernel_size=(1,1), stride=(1, 1))
        self.upsample2_conv_bn = nn.BatchNorm2d(128, track_running_stats=False, momentum=0.5)
        self.upsample3_conv  = nn.Conv2d(192, 128, kernel_size=(self.ks, self.ks), stride=(1, 1), padding=1)
        self.upsample3_conv_bn = nn.BatchNorm2d(128, track_running_stats=False, momentum=0.5)
        self.upsample4_conv  = nn.Conv2d(128, 128, kernel_size=(1,1), stride=(1, 1))
        self.upsample4_conv_bn = nn.BatchNorm2d(128, track_running_stats=False, momentum=0.5)
        self.upsample5_ConvTranspose2d = nn.ConvTranspose2d(128, 128, 3, stride=2, padding=1)
        self.upsample7_conv  = nn.Conv2d(128, 64, kernel_size=(1,1), stride=(1, 1))
        self.upsample7_conv_bn = nn.BatchNorm2d(64, track_running_stats=False, momentum=0.5)
        self.upsample8_conv  = nn.Conv2d(128, 64, kernel_size=(self.ks, self.ks), stride=(1, 1), padding=1)
        self.upsample8_conv_bn = nn.BatchNorm2d(64, track_running_stats=False, momentum=0.5)
        self.upsample9_conv  = nn.Conv2d(64, 64, kernel_size=(1,1), stride=(1, 1))
        self.upsample9_conv_bn = nn.BatchNorm2d(64, track_running_stats=False, momentum=0.5)
        self.upsample10_ConvTranspose2d = nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1)
        self.upsample11_conv  = nn.Conv2d(64, 64, kernel_size=(1,1), stride=(1, 1))
        self.upsample11_conv_bn = nn.BatchNorm2d(64, track_running_stats=False)
        self.upsample12_conv  = nn.Conv2d(80, 64, kernel_size=(self.ks, self.ks), stride=(1, 1), padding=1)
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
        

    def forward(self, x=torch.rand(1, 3, 640, 640), requires_grad=True):
        x = torch.from_numpy(x)
#        
#        x = Variable(x_111, requires_grad=requires_grad).cuda()
#        
        x = Variable(x, requires_grad=requires_grad).cuda()
        x_00 = x
        
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
        
        return indexing_pred, seg_pred
    
    def forward_eval(self, x= torch.randn(4, 3, 320, 320).cuda(), requires_grad=True):
    #        x = torch.from_numpy(x)
        with torch.no_grad(): 
            x_00 = x
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
            return indexing_output, indexing_argmax, seg_output, seg_argmax
#
#a=DentlyNet()
#a.eval()
#a.cuda()
#
#import time 
#start = time.time()
#indexing_output, indexing_argmax, seg_output, seg_argmax = a.forward_eval()
#print(time.time() - start)