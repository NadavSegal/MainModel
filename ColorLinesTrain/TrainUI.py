import csv

try:
	import Tkinter
	from Tkinter import ttk
except:
	import tkinter as Tkinter
	from tkinter import ttk
import cv2
import numpy as np
from TrainConfig import train_config
import os


from PIL import Image, ImageTk

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



# Configurations
# Enter static  path

def csv_to_dict(root_path='./log_creator/static/',
                csv_file_name='AI_Labels_1.csv', 
                skel_pred_ext='{}_skel.png', 
                skel_label_ext='{}_skellabel.png'):
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
#
#def get_list():
#	Images=[]
#	for ImageP in ImageDir:
#		for i in os.listdir(ImageP):
#			Image = os.path.join(ImageP,i)
#			ext = Image.split('.')[::-1][0].upper()
#			if ext in Extension:
#				Images.append(Image)
#	return Images

    

def load_image(mat_file_path):
    im_out = cv2.imread(mat_file_path)
    return im_out       


# Creating Canvas Widget
class PictureWindow(Tkinter.Canvas):
    def __init__(self, config_dict, *args, **kwargs):
        Tkinter.Canvas.__init__(self, *args, **kwargs)  
        self.config_dict = config_dict
        self.last_csv = self.config_dict['last_created_csv_file']
        self.csv_dict, self.OptionList= csv_to_dict( self.config_dict['created_static_path'], self.config_dict['last_created_csv_file'])
        self.imagelist_p=[]
        self.c_key = "none"
        self.c_im = None
        self.c_gt = None
        self.c_ai = None
        self.c_path = None
        self.all_function_trigger()
        
        
    def ComboboxSelectedFunc(self, *args):
        temp = self.opt.get()
        if temp == self.c_key:
            return
        else:  
            self.c_key = self.opt.get()
		    
#		load_image( c_path + ".png"), load_image(c), load_mat_file_as_np(c_path[:-8] + name + "_skelSubP.mat")
        if self.c_key in self.OptionList:
            r_path = self.csv_dict[self.c_key]["image_path"]
            image_path = r_path + ".png".upper()

            ai = self.csv_dict[self.c_key]["skel_pred_path"]
            gt = self.csv_dict[self.c_key]["skel_label_path"]
            
            self.c_path = r_path
            self.c_im = load_image(image_path)
            self.c_gt = load_image(gt)
            self.c_ai = load_image(ai)
            self.show_image(self.c_im, self.c_key)
            self.itemconfig(self.current_csv_file, text="current_csv_file: {0}".format(self.last_csv))
            self.itemconfig(self.idxAI_eqq_idxGT, text="idxAI_eqq_idxGT: {0}".format(self.csv_dict[self.c_key]['idxAI_eqq_idxGT']))
            self.itemconfig(self.err_mad, text="err_mad: {0}".format(self.csv_dict[self.c_key]['err_mad']))
            self.itemconfig(self.err_mean, text="err_mean: {0}".format(self.csv_dict[self.c_key]['err_mean']))
            self.itemconfig(self.correct_pixels, text="correct_pixels: {0}".format(self.csv_dict[self.c_key]['correct_pixels']))
            self.itemconfig(self.wrong_pixels, text="wrong_pixels: {0}".format(self.csv_dict[self.c_key]['wrong_pixels']))
            self.itemconfig(self.tested_pixels, text="tested_pixels: {0}".format(self.csv_dict[self.c_key]['tested_pixels']))
            self.itemconfig(self.accuracy, text="accuracy: {0}".format(self.csv_dict[self.c_key]['accuracy']))
                                                      
            
    def show_src(self):
	    self.show_image(self.c_im, self.c_path)
        
    def show_s_gt(self):
	    img = np.where(self.c_gt > 0, 0, self.c_im)
	    self.show_image(img, self.c_path)
        
    def show_s_ai(self):
	    img = np.where(self.c_ai > 0, 0, self.c_im)
	    self.show_image(img, self.c_path)
        
    def show_gt_vs_ai(self):
	    img = self.c_im.copy()
	    img[self.c_gt > 0]= 0
	    img[self.c_ai > 0]= 255
#	    img1 = np.where(self.c_gt > 0, 255, self.c_im)
        
	    self.show_image(img, self.c_path)
        
        
    def refresh(self):
        self.last_csv = find_last_csv(static_folder_path)
        self.old_csv = self.config_dict["last_created_csv_file"]
        if self.last_csv == self.old_csv:
            print("already updated")
            
        else:
            self.config_dict["last_created_csv_file"] = find_last_csv(self.config_dict["created_static_path"])
            print("update csv from {0} to {1}".format(self.old_csv, self.last_csv))
        
        
        
    def show_image(self, img, image_path):
        img= tk_image(img,self.winfo_screenwidth(),self.winfo_screenheight())
        self.delete(self.find_withtag("bacl"))
        self.allready=self.create_image(640.0, 450.0,image=img, anchor='center', tag="bacl")
	
        self.image=img
        print (self.find_withtag("bacl"))
        self.master.title("Image Viewer ({})".format(image_path))
        return

    def previous_image(self):
        try:
            pop = self.imagelist_p.pop()
            self.show_image(pop)
            self.imagelist.append(pop)
        except:
            pass
        return

    def next_image(self):
        try:
            pop = self.imagelist.pop()
		
            self.show_image(pop)
            self.imagelist_p.append(pop)
        except:
            pass
        return

    def all_function_trigger(self):
        self.create_buttons()
        self.window_settings()
        return

    def window_settings(self):
        self['width']=self.winfo_width()
        self['height']=self.winfo_height()
        return

    def create_buttons(self):
#		Tkinter.Button(self, text=" > ", command=self.next_image).place(x=(self.winfo_screenwidth()/1.1),y=(self.winfo_screenheight()/2))
#		Tkinter.Button(self, text=" < ", command=self.previous_image).place(x=20,y=(self.winfo_screenheight()/2))
        yt = 50
        Tkinter.Button(self, text="Src Img", command=self.show_src, height=2, width=10).place(x=400,y= yt)
        Tkinter.Button(self, text="Src + GT", command=self.show_s_gt, height=2, width=10).place(x=530,y= yt)
        Tkinter.Button(self, text="Src + AI", command=self.show_s_ai, height=2, width=10).place(x=660,y= yt)
        Tkinter.Button(self, text="GT vs AI", command=self.show_gt_vs_ai, height=2, width=10).place(x=790,y= yt)
        Tkinter.Button(self, text="Refresh", command=self.refresh, height=2, width=10).place(x=920,y= yt)
        self.opt = ttk.Combobox(self, values=self.OptionList, height=20, width=50)
        self.opt.place(x=640,y= 20, anchor='center')
        self.opt.bind("<<ComboboxSelected>>", self.ComboboxSelectedFunc)
        
        sx = 20
        sy = 150
        self.current_csv_file = self.create_text(10+ sx,-10 + sy, anchor="nw", text="current_csv_file: ")
        self.idxAI_eqq_idxGT = self.create_text(10+ sx,10+ sy, anchor="nw", text="idxAI_eqq_idxGT: ")
        self.err_mad = self.create_text(10+ sx,30+ sy, anchor="nw", text="err_mad: ")
        self.err_mean = self.create_text(10+ sx,50+ sy, anchor="nw", text="err_mean:")
        self.correct_pixels = self.create_text(10+ sx,90+ sy, anchor="nw", text="correct_pixels:")
        self.wrong_pixels = self.create_text(10+ sx,110+ sy, anchor="nw", text="wrong_pixels:")
        self.tested_pixels = self.create_text(10+ sx,130+ sy, anchor="nw", text="tested_pixels:")
        self.accuracy = self.create_text(10+ sx,150 + sy, anchor="nw", text="accuracy:") 

        self['bg']="white"
        return
    
    
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
    
    
    
    
# Main Function
def main(config_dict):
	# Creating Window
	root = Tkinter.Tk(className=" Image Viewer")
	root.config(height=800, width=1280)
	# Creating Canvas Widget
	PictureWindow(config_dict, root).pack(expand="yes",fill="both")
	# Not Resizable
	root.geometry("1280x800")

#	root.resizable(width=100,height=100)
	# Window Mainloop
	root.mainloop()
	return

# Image Extensions Allowed
UI_config = None
if __name__ == '__main__':
    static_folder_path = "{0}/{1}/static/".format(train_config['TrainingLogDir'], train_config['ModelConfig']['ModelName'])
    UI_config = {
                    "created_static_path": static_folder_path, 
                    "last_created_csv_file": find_last_csv(static_folder_path)
                 }
    
                 
    main(UI_config)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    