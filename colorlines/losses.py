import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
import numpy as np

class adversarial_own_loss(nn.Module):
    def __init__(self):
        super(adversarial_own_loss, self).__init__()
        
        self.device = torch.cuda.is_available()
        
        if self.device:
            self.dtype = torch.cuda.FloatTensor
            self.dtype_l = torch.cuda.LongTensor
        else:
            self.dtype = torch.FloatTensor
            self.dtype_l = torch.LongTensor
        
        
        self.adversarial_loss = torch.nn.BCELoss()
        self.adversarial_loss.cuda()
        
        w_class = np.ones(59)
        w_class[0] = 0.5
        w_class = torch.tensor(w_class).type(self.dtype)
        self.ce1 = nn.CrossEntropyLoss(w_class)
        
        w_class1 = np.ones(59)
        w_class1[0] = 0.5
        w_class1 = torch.tensor(w_class1).type(self.dtype)
        self.ce2 = nn.CrossEntropyLoss(w_class1)
        
        w_class3 = np.ones(2)
        w_class3 = torch.tensor(w_class3).type(self.dtype)
        self.ce3 = nn.CrossEntropyLoss(w_class3)

        
#        labels_one_hot = self.make_one_hot(y_labels_r, 49)

    def forward (self, macro_dr, macro_df, micro_dr, micro_df):
        
        
        
        valid_l = Variable(torch.ones((1, 1,3,3)), requires_grad=False).type(self.dtype)
        fake_l = Variable(torch.zeros((1, 1,3,3)), requires_grad=False).type(self.dtype)
        
        valid_h = Variable(torch.ones((1, 1,64,64)), requires_grad=False).type(self.dtype)
        fake_h = Variable(torch.zeros((1, 1,64,64)), requires_grad=False).type(self.dtype)
        
        
        real_l_loss = self.adversarial_loss(macro_dr, valid_l)
        fake_l_loss = self.adversarial_loss(macro_df, fake_l)
        
        real_h_loss = self.adversarial_loss(micro_dr, valid_h)
        fake_h_loss = self.adversarial_loss(micro_df, fake_h)
        
        d_loss =  torch.log(real_l_loss) + torch.log(torch.sub(torch.ones_like(fake_l_loss), fake_l_loss))
        d_loss1 =  torch.log(real_h_loss) + torch.log(torch.sub(torch.ones_like(fake_h_loss), fake_h_loss))
        
#        print(d_loss + d_loss1 )
#        loss2 = self.ce3(macro_dr, Variable(torch.ones_like(macro_dr[0]), requires_grad=False).type(torch.LongTensor))
#        loss3 = self.ce4(macro_df, Variable(torch.zeros_like(macro_dr[0]), requires_grad=False).type(torch.LongTensor))
#        
#        loss4 = self.ce5(micro_dr, torch.ones_like(micro_dr[0]).type(torch.LongTensor))
#        loss5 = self.ce6(micro_df, torch.zeros_like(micro_dr[0]).type(torch.LongTensor))
        
#        L_adver_l = torch.log(macro_dr) + torch.log(1 - macro_df)
#        L_adver_h = torch.log(micro_dr) + torch.log(1 - micro_df)
#        
##        loss = torch.sum(- labels_one_hot * F.log_softmax(logits_masked, 1), 1)
#        
#        o = L_adver_l + 25*loss1 + 1*L_adver_h +100*loss
#        o1 = torch.max(o)

        return d_loss**2 + d_loss1**2


class skeleton_based_loss(nn.Module):
    def __init__(self):
        super(skeleton_based_loss, self).__init__()
        
    def forward(self, x_output, y_labels):
                                
        y_labels = y_labels[..., None]
        y_labels = np.transpose(y_labels,(2, 0, 1))
        yy_t = Variable(torch.from_numpy(y_labels).float(), requires_grad=True).cuda()
        
        
        loss = torch.mean(torch.abs(x_output - yy_t))
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
        self.criterion.cuda()
        
        w_class1 = np.ones(49)
        w_class1[0] = 0.5
        w_class1 = torch.tensor(w_class1).type(self.dtype)
        self.criterion1 = nn.CrossEntropyLoss(w_class1)
        self.criterion1.cuda()
        
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

    def forward(self, x_output=torch.randn(1, 49, 641, 641).cuda(), y_labels=np.random.randint(0, 49, size=(1,1,641,641)), y1= np.random.randint(0, 49, size=(641,641))):
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