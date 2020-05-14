import numpy as np
import time
import importlib.util
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

#import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvas

def save_data_as_graph(data,
                       path_to_save="./log/1280_fast_6conv/static/my_img.png",
                       title='None'):
    x_steps = np.arange(len(data), dtype='int32')

    f = Figure(figsize=(5,5), dpi=100)
    canvas = FigureCanvas(f)
    a = f.add_subplot(111)
    a.set_title(title)
    a.set_xlabel('Train Step')
    a.set_ylabel('Loss')
    a.plot(x_steps.tolist(), data)

    canvas.print_figure(path_to_save)


def preprocess(self, x=None):
        x = torch.from_numpy(x).type(self.dtype)
        if self.ai.config['GPU']:
            x = x.cuda()
            
        x = x.permute(2,0,1)
        x = torch.div(x , 255.0)
        x = torch.unsqueeze(x, 0)
            
        return x
    
def to_torch_format(x, to_cuda=False):
    
    if to_cuda:
        #x = torch.from_numpy(x).type(torch.cuda.HalfTensor)
        x = torch.from_numpy(x).type(torch.cuda.HalfTensor)
        x = x.cuda()
    else:
        x = torch.from_numpy(x).type(torch.FloatTensor)
        
        
    shape = x.shape
    if len(shape) == 3:
        x = x.permute(2,0,1)
        x = torch.div(x , 255.0)
        x = torch.unsqueeze(x, 0)
    else:
        x = torch.unsqueeze(x, -1)
        x = x.permute(2,0,1)
        x = torch.unsqueeze(x, 0)
                      
    return x



def loadWeights(model, pt_full_path, useGPU):
    print("loading trained model from path: {0}".format(pt_full_path))
        
    model_dict = model.state_dict()
    if useGPU:
        last_state = torch.load(pt_full_path) 
    else:
        last_state = torch.load(pt_full_path, map_location={'cuda:0': 'cpu'})
            
    new_state = {k: v for k, v in last_state.items() \
                 if k in model_dict.keys()} # 1. filter out unnecessary keys        
    model_dict.update(new_state) # 2. overwrite entries in the existing state dict
    model.load_state_dict(model_dict)
    
    return model



def save_model(model, save_path, global_step):
    torch.save(model.state_dict(), '{0}/traind_model_{1}.pt'.format(save_path, global_step)) 
        
    return  '{0}/traind_model_{1}.pt'.format(save_path, global_step)

class Skeleton:
    def __init__(self, 
                 label, 
                 sub_label=None,
                 max_i=28,
                 min_i=1,
                 mode=1280):
        
        indices = (label).nonzero().squeeze()
        if len(indices) > 0:
        
            self.y = indices[:, 0].type(torch.LongTensor)
            self.x = indices[:, 1].type(torch.LongTensor)
            self.idxs = label[self.y, self.x].type(torch.IntTensor)
            
            self.idxs[self.idxs > max_i] = 0
            self.idxs[self.idxs <= min_i] = 0
            
            self.s_list = np.zeros((1, 64, mode))
            self.s_list[:,self.idxs, self.x] += self.y
            self.is_skel=True
        else:
            self.is_skel=False

def skel_score_postprocess(idx_per_channel_scores, use_cuda):
    dtype = None
    if use_cuda:
        dtype=torch.cuda.FloatTensor
    else:
        dtype=torch.FloatTensor
        
    pred_after_softmax = torch.exp(F.log_softmax(idx_per_channel_scores, 1))
    output, arg_max = pred_after_softmax.max(dim=1)
    a = output.shape
    a2 = arg_max.shape
    a3 = pred_after_softmax.shape
    #output = torch.where((arg_max != 0), output, torch.zeros(output.size(0), output.size(1), output.size(2)).type(dtype) - 50)
    output[arg_max == 0] = -50
    return output, arg_max
    

def CheckCuda():
    return torch.cuda.is_available()

def importModel(model_path, model_name=None, pyname=None):
#        model_path = self.config['ModelsPath']
#        model_name = self.config['ModelConfig']['ModelName']
        if not model_name is None:
            spec = importlib.util.spec_from_file_location("DentlyNet", "{0}/{1}/{2}".format(model_path, model_name, pyname))
        else:
            spec = importlib.util.spec_from_file_location("DentlyNet", "{0}".format(model_path))
            
        foo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(foo)
        return foo.DentlyNet()
    
def getOptimizer(modelParams, learningRate):
    #return torch.optim.Adam(modelParams, lr=learningRate)
    return torch.optim.SGD(modelParams, lr=learningRate)
#torch = importlib.import_module("torch")
#torch.set_num_threads(100)
class skeleton_based_loss(nn.Module):
    def __init__(self, use_cuda):
        super(skeleton_based_loss, self).__init__() 
        self.use_cuda = use_cuda
        
    def forward_OLD(self, x_output, y_labels, arg_max, yt1):
                                
        y_labels = y_labels[None, ...]
#        y_labels = np.transpose(y_labels,(2, 0, 1))
        yy_t = Variable(torch.from_numpy(y_labels.copy()), requires_grad=True).float()
        if self.use_cuda:
            yy_t = yy_t.cuda()
        
        
        #yt1 = torch.where((yt1 > 0), yt1, -5) #temp_zeros - 5)
        yt1[yt1 <= 0] = -5
#        yt1[yt1 == 0] = -5
        yt1 = torch.squeeze(yt1, 0)
        yt1 = Variable(yt1, requires_grad=True).float()
        if self.use_cuda:
            yt1 = yt1.cuda()
        
        
        arg_max = arg_max.float()
#        temp_zeros = torch.zeros(yt1.size(0), yt1.size(1), yt1.size(2))
#        if self.use_cuda:
#            temp_zeros = temp_zeros.cuda()
            

        
#        x_output[yt1 != arg_max] = -100
#        yy_t[(yt1 != arg_max) & (yt1 > 0)] = -100
        yy_t[(yt1 != arg_max) & (yt1 == -5) & (x_output > 0.5)] = 0
#        yy_t[(yt1 != arg_max) & (yt1 == -5) & (x_output > 0.5)] += x_output[(yt1 != arg_max) & (yt1 == -5) & (x_output > 0.5)] 
        loss = torch.mean(torch.abs(x_output[yy_t != 0] - yy_t[yy_t != 0]))
        return loss

    def forward(self, skeleton_scores, yy, y_t1): #skeleton only
        yy = yy[None, ...]
        yy_t = Variable(torch.from_numpy(yy.copy()), requires_grad=True).float()
        if self.use_cuda:
            yy_t = yy_t.cuda()

        y_t1 = torch.squeeze(y_t1, 0)
        y_t1 = Variable(y_t1, requires_grad=True).float()
        if self.use_cuda:
            y_t1 = y_t1.cuda()

        #skeleton_scores = skeleton_scores/torch.max(skeleton_scores)
        loss_1 = -torch.mean(skeleton_scores[0, yy_t > 0])
        loss_2 = torch.mean(skeleton_scores[0, yy_t == 0])
        loss = loss_1 + loss_2
        #text = torch.max(skeleton_scores)
        print(loss)
        return loss
    
class dont_care_crossentropy(nn.Module):
    def __init__(self):
        super(dont_care_crossentropy, self).__init__()

        self.device = torch.cuda.is_available()
        
        if self.device:
            #self.dtype = torch.cuda.FloatTensor
            self.dtype = torch.cuda.HalfTensor
            #self.dtype_l = torch.cuda.LongTensor
            self.dtype_l = torch.cuda.LongTensor
        else:
            self.dtype = torch.FloatTensor
            self.dtype_l = torch.LongTensor
            
        w_class1 = np.ones(28)
        w_class1 = [  0.        ,   0.        ,   0.        , 347.71392955,
        99.63892654,  39.37690214,  14.8825493 ,  12.39933117,
        12.90435531,  12.1819491 ,  14.86501946,  17.12773481,
        14.57515897,  16.20326133,  16.25254801,  16.38759622,
        19.09500656,  18.38026094,  18.31484437,  29.6057619 ,
        21.56770528,  34.27187323, 387.11045477, 407.52895782,
         0.        ,   0.        ,   0.        ,   0.        ]
        #w_class1[0] = 0
        w_class1 = torch.tensor(w_class1).type(self.dtype)
        self.criterion = nn.CrossEntropyLoss(w_class1, reduction = 'none')
        #self.criterion
            
    def forward(self, x_output=torch.randn(1, 28, 641, 641), y_labels=np.random.randint(0, 28, size=(1,1,641,641)), y1= np.random.randint(0, 28, size=(641,641))):

        y_labelsr = y1[None, ...]
        #y_labelsrWide = y_labelsr.copy()
        y_labelsr[0, 3:-3, :] = y_labelsr[0, 3:-3, :] + y_labelsr[0, 1:-5, :] + y_labelsr[0, 2:-4, :]\
                                + y_labelsr[0, 4:-2, :] + y_labelsr[0, 5:-1, :]
                                
        #y_labelsr[y_labelsr>56] = 0
        #y_labels_r = torch.from_numpy(y_labelsr).type(self.dtype_l)
        y_labels_r = torch.from_numpy(y_labelsr).type(self.dtype_l)

#        y_labels_r = y_labels_r + y_labels_r[]
#        y_labels[y_labels > 0] = 1
#        y_labels[y_labels <= 0] = 0
#        
#        y_labels_ = y_labels.clone().detach().requires_grad_(True).type(self.dtype)
#        
#        logits_masked = [x_output[:,i:i+1,:,:]*y_labels_ \
#                         for i in range(1, c)]
#        logits_masked.insert(0, x_output[:,0,:,:]*y_labels_)#torch.zeros_like(y_labels_))
#        
#        logits_masked = torch.cat(logits_masked, 1)
        y_labels_r[y_labels_r > 55] = 0 
        mask = y_labels_r > 0
        y_labels_r = torch.div(y_labels_r, 2)
        #y_labels_r = torch.remainder(y_labels_r, 28)
        #mask = mask.squeeze()
        
        loss = self.criterion(x_output, y_labels_r).float()/100000
        loss = torch.sum(loss[mask==True])
        #print(loss)
#        loss = torch.sum(- labels_one_hot * F.log_softmax(logits_masked, 1), 1)
        return loss
    
    
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







