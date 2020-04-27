import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
import UI_Utils as ui


LARGE_FONT= ("Verdana", 12)

class Page(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Color Line Train Exploration", font=LARGE_FONT)
        label.pack(pady=10,padx=10)
        self.controller = controller
        self.config_dict = controller.UI_config
        self.last_csv = self.config_dict['last_created_csv_file']
        self.csv_dict, self.OptionList= ui.csv_to_dict( self.config_dict['created_static_path'], self.config_dict['last_created_csv_file'])
        self.c_key = "none"
        self.c_im = None
        self.c_gt = None
        self.c_ai = None
        self.c_path = None
        self.colormap = ui.Index2Color(Index2Color_path="./Index2Color.txt")
        
#        self.canvas = tk.Canvas()
#        self.canvas.pack()
        buttons_dict = {}
        buttons_dict["Src  Img"] = self.show_src
        buttons_dict["Src + GT"] = self.show_src_with_gt
        buttons_dict["Src + AI"] = self.show_src_with_ai
        buttons_dict["GT vs AI"] = self.show_gt_vs_ai
        buttons_dict["inAI not inGT"] = self.inAI_not_inGT
        buttons_dict["inGT not inAI"] = self.inGT_not_inAI
        buttons_dict["Refresh"] = self.refresh
        
        tk.Button(self, 
                  text="<- MainMenu", 
                  command=lambda: controller.show_mainmenu(), 
                  height=2, 
                  width=10).place(relx=0.1,
                                  rely=0.03,
                                  anchor='center')
        
        
        
#        yt = 50
        nav_x_start = 200
        nav_y_start = 50
        nav_height=2
        nav_width = 10
        
        for name, function in buttons_dict.items():
            tk.Button(self, text=name, command=function, height=nav_height, width=nav_width).place(x=nav_x_start,y= nav_y_start)
            nav_x_start += 130
            
#        tk.Button(self, text="inAI not inGT", command=self.inAI_not_inGT, height=2, width=10).place(x=140,y= yt)
#        tk.Button(self, text="inGT not inAI", command=self.inGT_not_inAI, height=2, width=10).place(x=270,y= yt)
#        tk.Button(self, text="Src Img", command=self.show_src, height=2, width=10).place(x=400,y= yt)
#        tk.Button(self, text="Src + GT", command=self.show_src_and_gt, height=2, width=10).place(x=530,y= yt)
#        tk.Button(self, text="Src + AI", command=self.show_src_and_ai, height=2, width=10).place(x=660,y= yt)
#        tk.Button(self, text="GT vs AI", command=self.show_gt_vs_ai, height=2, width=10).place(x=790,y= yt)
#        tk.Button(self, text="Refresh", command=self.refresh, height=2, width=10).place(x=920,y= yt)
        self.opt = ttk.Combobox(self, values=self.OptionList, height=20, width=50)
        self.opt.place(x=640,y= 20, anchor='center')
        self.opt.bind("<<ComboboxSelected>>", self.ComboboxSelectedFunc)
        
        
        
        self.main_image = tk.Label(self, image ='')
        self.main_image.pack()
        self.main_image.place(relx=0.25, rely=0.15)
        
        self.skeleton_graph = tk.Label(self, image ='')
        self.skeleton_graph.pack()
        self.skeleton_graph.place(relx=0.8, rely=0.15)
        
        self.indexing_graph = tk.Label(self, image ='')
        self.indexing_graph.pack()
        self.indexing_graph.place(relx=0.8, rely=0.40)
        
        self.segmentation_graph = tk.Label(self, image ='')
        self.segmentation_graph.pack()
        self.segmentation_graph.place(relx=0.8, rely=0.65)
        
        self.log_dict = {}
        self.log_dict['current_csv_file'] = tk.Label(self, text="current_csv_file: ")
        self.log_dict['idxAI_eqq_idxGT'] = tk.Label(self, text="idxAI_eqq_idxGT: ")
        self.log_dict['err_mad'] = tk.Label(self, text="err_mad: ")
        self.log_dict['err_mean'] = tk.Label(self, text="err_mean: ")
        self.log_dict['correct_pixels'] = tk.Label(self, text="correct_pixels: ")
        self.log_dict['wrong_pixels'] = tk.Label(self, text="wrong_pixels: ")
        self.log_dict['tested_pixels'] = tk.Label(self, text="tested_pixels: ")
        self.log_dict['accuracy'] = tk.Label(self, text="accuracy: ")
        
        start_x = 30
        start_y = 140
        
        for key in self.log_dict:
            self.log_dict[key].place(x=start_x, y=start_y, anchor='nw')
            start_y += 20
            
        
        
    def show_image(self, img, image_path):
#        img = np.rot90(img)
        h, w = img.shape[:2]
        
        if h > 1280 or w > 800:
            img = cv2.resize(img, (int(w/2), int(h/2)))
        
        img= ui.tk_image(img,self.winfo_screenwidth(),self.winfo_screenheight())
#        img= tk_image(img,200, 200)
        self.main_image.config(image=img)

        self.image=img
        return
    
    
    def show_graph(self, img, graph_name=''):
        img = cv2.resize(img, (200, 200))
        img= ui.tk_image(img,200, 200)
        
        
        if graph_name == 'skeleton':
            self.skeleton_graph.config(image=img)
            self.skeleton=img
            
        if graph_name == 'indexing':
            self.indexing_graph.config(image=img)
            self.indexing=img
            
        if graph_name == 'segmentation':
            self.segmentation_graph.config(image=img)
            self.segmentation = img

        return
    
    def show_src(self):
	    self.show_image(self.src_image, self.c_path)
            
    def show_src_with_gt(self):
	    img = np.where(self.colormap.apply(self.gt_image) > 0, 0, self.src_image)
	    self.show_image(img, self.c_path)
        
    def show_src_with_ai(self):
	    img = np.where(self.colormap.apply(self.ai_image) > 0, 0, self.src_image)
	    self.show_image(img, self.c_path)
        
    def show_gt_vs_ai(self):
	    img = self.src_image.copy()
	    img[self.gt_image > 0]= 0
	    img[self.ai_image > 0]= 255

	    self.show_image(img, self.c_path)
        
    def inGT_not_inAI(self):
        img = self.src_image.copy()
        gt_mask = self.gt_image > 0
        ai_mask = self.ai_image == 0
        img[np.logical_and(gt_mask, ai_mask)]= 0

        self.show_image(img, self.c_path)
        
    def inAI_not_inGT(self):
        img = self.src_image.copy()
        gt_mask = self.gt_image == 0
        ai_mask = self.ai_image > 0
        img[np.logical_and(gt_mask, ai_mask)]= 0

        self.show_image(img, self.c_path)
    
    def refresh(self):
        self.last_csv = ui.find_last_csv(self.config_dict['created_static_path'])
        self.old_csv = self.config_dict["last_created_csv_file"]
        if self.last_csv == self.old_csv:
            print("already updated")
            
        else:
            self.config_dict["last_created_csv_file"] = ui.find_last_csv(self.config_dict["created_static_path"])
            print("update csv from {0} to {1}".format(self.old_csv, self.last_csv))
    
    def ComboboxSelectedFunc(self, *args):
        temp = self.opt.get()
        if temp == self.c_key:
            return
        else:  
            self.c_key = self.opt.get()
		    
#		load_image( c_path + ".png"), load_image(c), load_mat_file_as_np(c_path[:-8] + name + "_skelSubP.mat")
        if self.c_key in self.OptionList:
            r_path = self.csv_dict[self.c_key]["image_path"]
            image_path = r_path + ".png"

            ai = self.csv_dict[self.c_key]["skel_pred_path"]
            gt = self.csv_dict[self.c_key]["skel_label_path"]
            
            self.c_path     = r_path
            self.src_image  = ui.read_image(image_path)
            self.gt_image   = ui.read_image(gt)
            self.ai_image   = ui.read_image(ai)
            
            self.gt_image = self.gt_image
            self.ai_image = self.ai_image
            
            
            
            self.indexing_loss      = ui.read_image(self.config_dict['created_static_path'] + "idx_loss_all.png")
            self.skeleton_loss    = ui.read_image(self.config_dict['created_static_path'] + "seg_loss_all.png")
            self.segmentation_loss  = ui.read_image(self.config_dict['created_static_path'] + "skel_loss_all.png")
            
            
            self.show_image(self.src_image, self.c_key)
            self.show_graph(self.indexing_loss,     graph_name = 'indexing')
            self.show_graph(self.skeleton_loss,     graph_name = 'skeleton')
            self.show_graph(self.segmentation_loss, graph_name = 'segmentation')
            
            # show logs  
            self.log_dict['current_csv_file'].config(text="current_csv_file: {0}".format(self.last_csv))
            for log_key in self.csv_dict[self.c_key].keys():
                if log_key in self.log_dict:
                    self.log_dict[log_key].config(text="{0}: {1}".format(log_key, self.csv_dict[self.c_key][log_key]))
