from __future__ import print_function

import numpy as np
import time
import torch
import dentlyset 
import dently_model_t as dently_model


def get_matrix_accuracy(procces_frame_output, src_labels):
    procces_frame_output[src_labels <= 0] = -1
        
    true_mask = procces_frame_output == src_labels
        
    true_mask = np.sum(true_mask) / np.sum(src_labels>0)
        
    return true_mask, 1 - true_mask

def img_and_labels_to_model_input_format(x=None, y=None):
        
        if not x is None:
            x = np.transpose(x,(2,0,1))
            x = x.astype('float32') 
            x = x / 255.0
            x = x[None,...]
            
        if not y is None:
            y = y[..., None]
            y = np.transpose(y,(2, 0, 1))
            y = y[None,...]
#            y = np.swapaxes(y, -1, 1)
                
        return x, y
    
    
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

class Eval:
    def __init__(self):        
        self.model  = dently_model.DentlyNet()
        self.model.eval()
        self.mp2d = torch.nn.MaxPool2d(kernel_size = (7,1), stride=(1, 1), padding=(3,0))
        self.gpu=torch.cuda.is_available()
        if self.gpu:
            self.dtype = torch.cuda.FloatTensor
            self.dtype_l = torch.cuda.LongTensor
        else:
            self.dtype = torch.FloatTensor
            self.dtype_l = torch.LongTensor
        
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

#                self.model.load_state_dict(model_dict)
                self.model.load_state_dict(torch.load(pt_trained_model_path))
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
    
    def from_path_for_socket(self, real_image):   
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

    
    def from_path(self, x_path):            
        real_image = dentlyset.load_image(x_path)
        #real_image = cv2.cvtColor(real_image, cv2.COLOR_BGR2RGB)
#        temp1 = np.zeros((1280,1280, 3))
#        s1 = np.random.randint(0,630)
#        s2 = np.random.randint(0,630)
#        
#        temp1[s1:s1 + 640, s2:s2+640, :] += real_image.copy()
#        real_image = temp1
#        
        x = self.img_to_input_format(real_image)
        indexing_output, indexing_argmax, seg_output, seg_argmax, sk_x = self.model.forward_eval(x)
        indexing_output, indexing_argmax = self._process_indexing(indexing_output, indexing_argmax)
        
        skeleton = self._process_skelaton(indexing_output, indexing_argmax)
        with torch.no_grad():
#            torch.cuda.synchronize()
            skeleton_o = skeleton.cpu().data.numpy()

        indexing_output = indexing_output.cpu().detach().numpy()  
        indexing_argmax = indexing_argmax.cpu().detach().numpy()
        
        return real_image, skeleton_o, indexing_output, indexing_argmax, seg_output, seg_argmax, sk_x#, s1, s2   
    
    def _predict(self, 
                   x):
        
        x_, _ = img_and_labels_to_model_input_format(x)
        indexing_pred, seg_pred = self.model.forward_eval(x_)
        
        return x, indexing_pred, seg_pred
    
    def _draw_results(self,
                      src_image,
                      mask):
        
        b = mask[..., None] > 0
        bbb = np.concatenate([b, b, b], axis=2)
        
        src_image[bbb > 0] = 0
        
        return src_image
    
    def _process_indexing(self,
                          indexing_output, 
                          indexing_argmax,
                          th=0.5):
        
        value_max = indexing_output
        value_max[value_max < th] = 0
        indexing_output[value_max == 0] = 0
        indexing_argmax[value_max == 0] = 0

        return indexing_output, indexing_argmax
    
    
    
    def _process_skelaton(self, process_indexing, indexing_argmax):
        

        test11 = self.mp2d(process_indexing.type(self.dtype))
        process_indexing = torch.where((indexing_argmax != 0), process_indexing.type(self.dtype), torch.FloatTensor(process_indexing.size(0),  process_indexing.size(1), process_indexing.size(2)).zero_().type(self.dtype))

        indexing_output = torch.where((process_indexing == test11), indexing_argmax.type(self.dtype), torch.FloatTensor(process_indexing.size(0),  process_indexing.size(1), process_indexing.size(2)).zero_().type(self.dtype))

        return indexing_output
    

# eval steps example
#a = Eval()
## a.load_Weights('C://sw//demo_algo//sources//AI//traind_model_37500_320_Tooth10_WithoutBlood_z.pt')
##a.load_Weights('C://sw//demo_algo//sources//AI//traind_model_42000_Tooth10_WithoutBlood_z.pt')
#a.load_Weights('/home/alon/dently_7_19/30_7_19/log/traind_model_170000.pt')
#p ='/media/alon/SP PHD U3/AI_Ref/Rar/F__Records__ProtoTypeV2__Invitro__S_12.6.2__3-Tooth9_S_12.6.2_calibfile070319_20190310/00000448.png' #00000865 00000565 000001025 bug in 00002005
#pp ='/media/alon/SP PHD U3/AI_Ref/Rar/F__Records__ProtoTypeV2__Invitro__S_11.3.3__1-MartinTooth44_S_11.3.3_20190307_scanZ/00000565.png'
#pl ='/media/alon/SP PHD U3/AI_Ref/Rar/F__Records__ProtoTypeV2__Invitro__S_12.6.2__3-Tooth9_S_12.6.2_calibfile070319_20190310/00000448_indRef.png'
#y_src = cv2.imread(pl)
#y_src = y_src[:, :, 0]
##p = 'D:\TempData\Tooth10_WithoutBlood_z\log_AI_640/temp.bmp'
#real_image, skeleton, indexing_output, indexing_argmax, seg_output, seg_argmax = a.from_path(p)
## print("1")
#vis_indexing = a._draw_results(real_image.copy(), skeleton[0])
#real_indexing = a._draw_results(real_image.copy(), y_src)
## plt.imshow(vis_indexing)
#
#
## plt.matshow(vis_indexing.astype("uint8"));
## plt.colorbar()
## plt.show()
#
#cv2.imshow("pred_indexing",  vis_indexing.astype("uint8"))
#cv2.imshow("label_indexing",  real_indexing.astype("uint8"))
#cv2.waitKey(0)


# seg steps example
#x_seg, indexing_pred, seg_pred = a._predict(x1)
#seg_output, seg_argmax = postprocces_pred(seg_pred)
#process_seg = a._process_seg(seg_output, seg_argmax)
#segvis_image = a._draw_results(x_seg, process_seg)
#cv2.imshow("segvis_image", segvis_image.astype("uint8"))
#cv2.waitKey(0)


