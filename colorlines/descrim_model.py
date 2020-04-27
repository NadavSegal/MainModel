import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
#from losses import dont_care_crossentropy
import numpy as np


class D_net(nn.Module):
    def __init__(self, input_x=0, output_channels=49, is_training=True, batch_size=1):
        super(D_net, self).__init__()
        self.input_x = input_x
        self.batch_size=batch_size
        self.output_channels = output_channels
        self.ks = 3
        self.device = torch.cuda.is_available()
        

        self.macro_d_64_8_8 = nn.Conv2d(output_channels, 64, kernel_size=(4,4), stride=(2, 2), padding=3)
        self.macro_d_64_8_8_bn = nn.BatchNorm2d(64, track_running_stats=False, momentum=0.5)
        
        self.macro_d_128_4_4 = nn.Conv2d(64, 128, kernel_size=(4,4), stride=(2, 2))
        self.macro_d_128_4_4_bn = nn.BatchNorm2d(128, track_running_stats=False, momentum=0.5)
        
        self.macro_d_256_1_1 = nn.Conv2d(128, 256, kernel_size=(4,4),stride=2)
        
        self.macro_d_1_1_1 = nn.Conv2d(256, 1, kernel_size=1,stride=1)
        self.macro_d_1_1_1_bn = nn.BatchNorm2d(1, track_running_stats=False, momentum=0.5)
        
        
        
        self.micro_d_64_128_128 = nn.Conv2d(output_channels, 64, kernel_size=(4,4), stride=(2, 2), padding=1)
        self.micro_d_64_128_128_bn = nn.BatchNorm2d(64, track_running_stats=False, momentum=0.5)
        
        self.micro_d_128_64_64 = nn.Conv2d(64, 128, kernel_size=(4,4), stride=(2, 2), padding=2)
        self.micro_d_128_64_64_bn = nn.BatchNorm2d(128, track_running_stats=False, momentum=0.5)
        
        self.micro_d_1_64_64 = nn.Conv2d(128, 1, kernel_size=(4,4), padding=1)
        self.micro_d_1_64_64_bn = nn.BatchNorm2d(1, track_running_stats=False, momentum=0.5)       
        if self.device:
            self.dtype = torch.cuda.FloatTensor
            self.dtype_l = torch.cuda.LongTensor
        else:
            self.dtype = torch.FloatTensor
            self.dtype_l = torch.LongTensor
            
            
#        self.loss = dont_care_crossentropy()
        

    def forward(self, x=torch.randn(1, 49, 640, 640), y=torch.randn(1, 49, 640, 640), requires_grad=True):
        macro_dr = nn.functional.interpolate(y, size=(32, 32), mode='bilinear', align_corners=True )
        
#        macro_dr = torch.cat((x_src_low_res, src_low_labels), 1)
        macro_dr = F.leaky_relu(self.macro_d_64_8_8_bn(self.macro_d_64_8_8(macro_dr)))
        macro_dr = F.leaky_relu(self.macro_d_128_4_4_bn(self.macro_d_128_4_4(macro_dr)))
        macro_dr =  F.leaky_relu(self.macro_d_256_1_1 (macro_dr))
        macro_dr = self.macro_d_1_1_1(macro_dr)
        macro_dr = macro_dr**2 / (macro_dr**2 + 1)
        
        macro_df = nn.functional.interpolate(x, size=(32, 32), mode='bilinear', align_corners=True )
        macro_df = F.leaky_relu(self.macro_d_64_8_8_bn(self.macro_d_64_8_8(macro_df)))
        macro_df = F.leaky_relu(self.macro_d_128_4_4_bn(self.macro_d_128_4_4(macro_df)))
        macro_df =  F.leaky_relu(self.macro_d_256_1_1 (macro_df))
        macro_df = self.macro_d_1_1_1(macro_df)
        macro_df = macro_df**2 / (macro_df**2 + 1)
        
        micro_dr = nn.functional.interpolate(y, size=(256, 256), mode='bilinear', align_corners=True )
        micro_dr = F.leaky_relu(self.micro_d_64_128_128_bn(self.micro_d_64_128_128(micro_dr)))
        micro_dr = F.leaky_relu(self.micro_d_128_64_64_bn(self.micro_d_128_64_64(micro_dr)))
        micro_dr = self.micro_d_1_64_64(micro_dr)
        micro_dr = micro_dr**2 
        micro_dr  = micro_dr / (micro_dr.max()+ 1)
        
        micro_df = nn.functional.interpolate(x, size=(256, 256), mode='bilinear', align_corners=True )
        micro_df = F.leaky_relu(self.micro_d_64_128_128_bn(self.micro_d_64_128_128(micro_df)))
        micro_df = F.leaky_relu(self.micro_d_128_64_64_bn(self.micro_d_128_64_64(micro_df)))
        micro_df = self.micro_d_1_64_64(micro_df)
        micro_df = micro_df**2 
        micro_df = micro_df / (micro_df.max()+ 1)
#        print(macro_dr,macro_df, micro_dr, micro_df)
        return macro_dr, macro_df, micro_dr, micro_df
    

#a=D_net()
#macro_dr, macro_df, micro_dr, micro_df = a()
#a.cuda()
#
#import time 
#start = time.time()
#indexing_output, indexing_argmax, seg_output, seg_argmax = a.forward_eval()
#print(time.time() - start)