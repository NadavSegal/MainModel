import csv

import cv2
import numpy as np
import os
import torch 
import re

from PIL import Image, ImageTk

import matplotlib.pyplot as plt
import TrainUtils





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
    def __init__(self, model_py_full_path):        
        self.model  = TrainUtils.importModel(model_py_full_path)
        self.model.eval()
        self.mp2d = torch.nn.MaxPool2d(kernel_size = (13,1), stride=(1, 1), padding=(6,0))
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
        x = torch.from_numpy(x.copy()).type(self.dtype)
        x = x.permute(2,0,1)
        x = torch.div(x , 255.0)
        x = torch.unsqueeze(x, 0)
            
        return x
    
    
    def from_image(self, real_image):            
#        src_HLS = cv2.cvtColor(real_image, cv2.COLOR_BGR2HLS)
#        real_image = np.concatenate([real_image, src_HLS],axis=2)
        x = self.img_to_input_format(real_image)
        #indexing_output, indexing_argmax, seg_output, seg_argmax, sk_x = self.model.forward(x)
        #final_res, skeleton_final, skeleton_pred, indexing_pred, indexing_argmax = self.model.forward(x)
        final_res, skeleton_final, skeleton_pred, indexing_tensor, indexing_argmax, indexing_pred = self.model.forward(x)
        #indexing_output, indexing_argmax = self._process_indexing(indexing_output, indexing_argmax)
        
        #skeleton = self._process_skelaton(indexing_pred, indexing_argmax)
        with torch.no_grad():
#            torch.cuda.synchronize()
            #skeleton_o = skeleton.cpu().data.numpy()
            final_res = final_res.cpu().data.numpy()
            skeleton_pred = skeleton_pred.cpu().data.numpy()
            indexing_pred = indexing_pred.cpu().data.numpy()
            skeleton_final = skeleton_final.cpu().data.numpy()
        final_res = 2*final_res[0].copy()
        skeleton_pred = skeleton_pred[0, 0].copy()
        skeleton_final = skeleton_final[0, 0].copy()
        indexing_pred = indexing_pred[0].copy()
        final_res[indexing_pred < 0.9] = 0
        final_res[skeleton_final < 1] = 0
        #final_res[skeleton_pred < 2] = 0
        #np.sum(indexing_pred>0.99999)/np.sum(indexing_pred >-3)

#        indexing_output = indexing_output.cpu().detach().numpy()  
#        indexing_argmax = indexing_argmax.cpu().detach().numpy()
#        seg_output = seg_output.cpu().detach().numpy()  
#        seg_argmax = seg_argmax.cpu().detach().numpy()
        
        return final_res
    
    def _predict(self, x):
        
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
        
        value_max = indexing_output.clone().detach()
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





ROOT_DATASET_PATH = '/Users/alontetro/Downloads/AI_Labels_Shai_RT-2/'

class indexing_histogram:
    def __init__(self, 
                 min_index=2, 
                 max_index=49):
        
        self.indexing_hist = None #np.zeros((max_index + 1))
        self.n_bins = max_index + 1
        self.range = (min_index, max_index + 1)
    
    def single_apply(self, 
                     indexing,
                     img_name='./test111.png'):
        
        signal = indexing[np.nonzero(indexing)]
        
        #hist, bin_edges = np.histogram(indexs, bins=n_bins)
        
        fig, ax = plt.subplots()   
        hist, bins, _ = ax.hist(signal,  
                                bins=self.n_bins, 
                                range=self.range, 
                                density=False, 
                                facecolor='g', 
                                alpha=0.75)
#        fig.canvas.draw()

# grab the pixel buffer and dump it into a numpy array

        fig.savefig(img_name)
        hist_image = np.array(fig.canvas.renderer.buffer_rgba())
        hist_image = hist_image[:,:,:3]
        plt.close(fig)
        
        return hist, bins, hist_image
    
    def multiple_apply(self,
                  indexing_list,
                  pattern="_indRef.png"):
        
        for sample in indexing_list:
            print(sample)
            
            
            

DEFULT_SHAPE=(640, 640, 3)

def read_image(image_path):
    im_out = cv2.imread(image_path)
    return im_out 


def try_read_image(image_path):
    valid_path = False
    if os.path.isfile(image_path):
        im_out = cv2.imread(image_path)
        valid_path = True
    else:
        im_out = np.zeros(DEFULT_SHAPE, dtype="uint8")
        
    return im_out, valid_path


def read_Index2Color_file(Index2Color_path="./Index2Color.txt"):
    out_list = []
    fh = open(Index2Color_path)
    for line in fh:
        try:
            num = int(line)
            out_list.append(num)  
        except:
            continue
        
    fh.close()
    return out_list

def get_indexing_list(min_idx=2, 
                      max_idx=49, 
                      even=True):
    
    out = None
    if even:
        out = np.arange(min_idx, max_idx, 2)
    else:
        out = np.arange(min_idx, max_idx, 1)
        
    return out 
        

class Index2Color:
    def __init__(self, 
                 Index2Color_path="./Index2Color.txt"):
        self.color_dict = {1: [255, 0, 0], 
                          2: [0, 255, 0],
                          3: [0, 0, 255],
                          4: [0, 255, 255],
                          5: [255, 0, 255],
                          6: [255, 255, 0],
                          7: [255, 255, 255],
                          8: [64, 64, 64]}

        self.Index2Color = read_Index2Color_file(Index2Color_path)
        self.indexing = get_indexing_list()
        
    def apply_old(self, skeleton):
        
        out = np.zeros_like(skeleton)
        
        for index in self.indexing:
#            index = int(index)
            new_color = self.color_dict[self.Index2Color[index//2]]
            if np.sum(skeleton == index) > 0 and new_color != 0:
                out[np.where((skeleton==[index,index,index]).all(axis=2))] = self.color_dict[self.Index2Color[index//2]]
        
        return out
    
    def apply(self, skeleton):
        
        out = np.zeros_like(skeleton)
        label = skeleton[:, :, 0]
        
        x, y = label.nonzero()
        values = label[x, y]
        for i, val in enumerate(values):
        #    print()
            sss = val //2 
            if sss < len(self.Index2Color):
                t11 = self.Index2Color[sss]
                if t11 in self.color_dict:
                    out[x[i], y[i]] =  self.color_dict[t11]
                    
        return out
        
        
        
#aa = Index2Color()       
#label = read_image('/Users/alontetro/Downloads/AI_Labels/00003558_indRef.png')          
#out = aa.apply(label)
#cv2.imshow("out", out)
#cv2.waitKey(0)


def search_for_data_in_dir(path, search_pattern=r"[0-9]+.png"):
    out = []
    if not os.path.isdir(path):
        return out.append("cant_find_any_relevant_file")
    
    for file_name in os.listdir(path):
            if re.match(search_pattern, file_name):
                out.append(file_name)
                print(file_name)
                
    return out
    

def tk_image(img,w,h):
#    img = cv2.imread(path)
    # You may need to convert the color.
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    
#	img = Image.open(path)
#	img = img.resize((w,h))
    storeobj = ImageTk.PhotoImage(img)
    return storeobj


def csv_to_dict(root_path='./log_creator/static/',
                csv_file_name='AI_Labels_1.csv', 
                skel_pred_ext='{}_skel.png', 
                skel_label_ext='{}_skellabel.png'):
    try:
        out_dict = {}
        with open(root_path + csv_file_name, newline='') as csvfile:
             reader = csv.DictReader(csvfile)
             for row in reader:
                 image_number = row['image_path'].split("/")[-1]
                 row['skel_pred_path'] = root_path + skel_pred_ext.format(image_number)
                 row['skel_label_path'] = root_path + skel_label_ext.format(image_number)
                 out_dict[image_number] = row
        
        out_keys = list(out_dict.keys())
        out_keys.sort()
        
        return out_dict, out_keys
    
    except:
        return {}, []


def duplicate_label(label):
    out = label.copy()
    temp = label.copy()
    
    out[:-1, :] += temp[1:,:]
    out[:-2, :] += temp[2:,:]
    out[:-3, :] += temp[3:,:]
    out[1:, :] += temp[:-1,:]
    out[2:, :] += temp[:-2,:]
    out[3:, :] += temp[:-3,:]
    
    return out
    
    
def find_last_csv(path_to_search):
    last_csv = None
    max_idx = 0
    for path in os.listdir(path_to_search):
        if "csv" in path:
            number_str = path.split(".csv")[0].split("_")[1]
            if int(number_str) > max_idx:
                max_idx = int(number_str) 
                last_csv = path
                
    return last_csv
#        print(path)
    
