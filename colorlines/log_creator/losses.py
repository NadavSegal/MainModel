import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
import numpy as np

class adversarial_own_loss(nn.Module):
    def __init__(self):
        super(adversarial_own_loss, self).__init__() 
        self.adversarial_loss = torch.nn.BCELoss()
        
        
#        labels_one_hot = self.make_one_hot(y_labels_r, 49)

    def forward (self, real, fake):
        v = Variable(torch.ones((1, 1)), requires_grad=False).type(torch.cuda.FloatTensor)
        f = Variable(torch.zeros((1, 1)), requires_grad=False).type(torch.cuda.FloatTensor)
        
        ad_loss1 = self.adversarial_loss(real, v)
        ad_loss2 = self.adversarial_loss(fake, f)
    
        return ad_loss1 + ad_loss2


class skeleton_based_loss(nn.Module):
    def __init__(self):
        super(skeleton_based_loss, self).__init__() 
        
    def forward(self, x_output, y_labels):
                                
        y_labels = y_labels[..., None]
        y_labels = np.transpose(y_labels,(2, 0, 1))
        yy_t = Variable(torch.from_numpy(y_labels).float(), requires_grad=True)#.cuda()
        
        
        loss = torch.mean(torch.abs(x_output - yy_t))
        return loss
    
    
class skeleton_based_loss1(nn.Module):
    def __init__(self):
        super(skeleton_based_loss1, self).__init__() 
        
    def forward(self, x_output, y_labels):
        yy_t = Variable(torch.from_numpy(y_labels).float(), requires_grad=True)#.cuda()
        loss = torch.mean(torch.abs(x_output[yy_t > 0] - yy_t[yy_t > 0]))
        
        return loss

class dont_care_crossentropy(nn.Module):
    def __init__(self):
        super(dont_care_crossentropy, self).__init__()

        self.device = torch.cuda.is_available()
        
        if self.device:
            self.dtype = torch.cuda.FloatTensor
            self.dtype_l = torch.cuda.LongTensor
        else:
            self.dtype = torch.FloatTensor
            self.dtype_l = torch.LongTensor
#            
            
        w_class = np.ones(3)
        w_class[0] = 0.5
        w_class = torch.tensor(w_class).type(self.dtype)
        self.criterion = nn.CrossEntropyLoss(w_class)
        self.criterion
        
        w_class1 = np.ones(49)
        w_class1[0] = 0.5
        w_class1 = torch.tensor(w_class1).type(self.dtype)
        self.criterion1 = nn.CrossEntropyLoss(w_class1)
        self.criterion
        
    def make_one_hot(self, labels, C=2):
        '''
        Converts an integer label torch.autograd.Variable to a one-hot Variable.
        
        Parameters
        ----------
        labels : torch.autograd.Variable of torch.cuda.LongTensor
            N x 1 x H x W, where N is batch size. 
            Each value is an integer representing correct classification.
        C : integer. 
            number of classes in labels.
        
        Returns
        -------
        target : torch.autograd.Variable of torch.cuda.FloatTensor
            N x C x H x W, where C is class number. One-hot encoded.
        '''
        one_hot = torch.FloatTensor(labels.size(0), C, labels.size(2), labels.size(3)).zero_()
        one_hot = [torch.where((labels == i), torch.ones(labels.size(0), 1, labels.size(2), labels.size(3)).type(self.dtype), torch.FloatTensor(labels.size(0), 1, labels.size(2), labels.size(3)).zero_().type(self.dtype)) \
                         for i in range(C)]
        target = torch.cat(one_hot[:], 1)
        
        target = Variable(target, requires_grad=True)
            
        return target      

    def forward(self, x_output=torch.randn(1, 49, 641, 641), y_labels=np.random.randint(0, 49, size=(1,1,641,641)), y1= np.random.randint(0, 49, size=(641,641))):
#        y_labels[y_labels <= 0] = -1
        c = x_output.shape[1]
        y_labelsr = y1[None, ...]
        y_labels_r = torch.from_numpy(y_labelsr).type(self.dtype_l)
        y_labels[y_labels > 0] = 1
        y_labels[y_labels <= 0] = 0
        
        y_labels_ = Variable(torch.tensor(y_labels).type(self.dtype))
        
        logits_masked = [x_output[:,i:i+1,:,:]*y_labels_ \
                         for i in range(1, c)]
        logits_masked.insert(0, x_output[:,0,:,:]*y_labels_)#torch.zeros_like(y_labels_))
        
        logits_masked = torch.cat(logits_masked, 1)
        
#        labels_one_hot = self.make_one_hot(y_labels_r, 49)

        
        if c == 3:
            loss = self.criterion(logits_masked, y_labels_r)
        else:
            loss = self.criterion1(logits_masked, y_labels_r)
#        loss = torch.sum(- labels_one_hot * F.log_softmax(logits_masked, 1), 1)
        return loss
    


#loss = dont_care_crossentropy()
#loss.cuda()
#print(loss.forward())