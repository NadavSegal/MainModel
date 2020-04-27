import tkinter as tk
from tkinter import ttk, filedialog
import cv2
import numpy as np
import UI_Utils as ui
from log_creator import test_summary


LARGE_FONT= ("Verdana", 12)

class Page(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Color Line Train Exploration", font=LARGE_FONT)
        label.pack(pady=10,padx=10)
        
        self.controller = controller
        self.config_dict = {'current_root_path': None}
        self.eval = None #ui.Eval()
        self.trainedpt_path = None
        self.model_loaded = False
#        self.last_csv = self.config_dict['last_created_csv_file']
#        self.csv_dict, self.OptionList= ui.csv_to_dict( self.config_dict['created_static_path'], self.config_dict['last_created_csv_file'])
#        self.c_key = "none"
        self.c_im = None
        self.c_gt = None
        self.c_ai = None
        self.c_path = None
        self.colormap = ui.Index2Color(Index2Color_path="./Index2Color.txt")
        self.indexing_histogram = ui.indexing_histogram(min_index=2, 
                                                        max_index=49)
        
#        self.canvas = tk.Canvas()
#        self.canvas.pack()
        self.func_buttons_dict = {}
        self.buttons_dict = {}
        self.config_buttons = {}
        
        self.config_buttons["test dir"] = {"func": self.choose_root,
                                    "text": tk.StringVar(),
                                    "height": 2, 
                                    "width": 50, 
                                    'relx': 0.5,
                                    'rely': 0.03,
                                    'anchor': 'center', 
                                    'show': True,
                                    'name': 'test dir'}
        
        self.config_buttons["test dir"]['button'] = tk.Button(self, 
                                      text=self.config_buttons["test dir"]['text'], 
                                      command=self.config_buttons["test dir"]['func'], 
                                      height=self.config_buttons["test dir"]['height'], 
                                      width=self.config_buttons["test dir"]['width'])
        self.config_buttons["test dir"]['button'].place(relx=self.config_buttons["test dir"]['relx'],
                                                      rely=self.config_buttons["test dir"]['rely'],
                                                      anchor=self.config_buttons["test dir"]['anchor']
                                                      )
        self.config_buttons["test dir"]['button'].config(text="Choose working dir: ")
        
        
        tk.Button(self, 
                  text="<- MainMenu", 
                  command=lambda: controller.show_mainmenu(), 
                  height=2, 
                  width=10).place(relx=0.1,
                                  rely=0.03,
                                  anchor='center')
        
    
#        self.opt = ttk.Combobox(self, values=self.OptionList, height=20, width=50)
#        self.opt.place(x=640,y= 20, anchor='center')
#        self.opt.bind("<<ComboboxSelected>>", self.ComboboxSelectedFunc)
        
        
        
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
        
#        self.log_dict = {}
#        self.log_dict['current_csv_file'] = tk.Label(self, text="current_csv_file: ")
#        self.log_dict['idxAI_eqq_idxGT'] = tk.Label(self, text="idxAI_eqq_idxGT: ")
#        self.log_dict['err_mad'] = tk.Label(self, text="err_mad: ")
#        self.log_dict['err_mean'] = tk.Label(self, text="err_mean: ")
#        self.log_dict['correct_pixels'] = tk.Label(self, text="correct_pixels: ")
#        self.log_dict['wrong_pixels'] = tk.Label(self, text="wrong_pixels: ")
#        self.log_dict['tested_pixels'] = tk.Label(self, text="tested_pixels: ")
#        self.log_dict['accuracy'] = tk.Label(self, text="accuracy: ")
#        
#        start_x = 30
#        start_y = 140
#        
#        for key in self.log_dict:
#            self.log_dict[key].place(x=start_x, y=start_y, anchor='nw')
#            start_y += 20
            
    def init_exploration(self):
#        self.config_dict = {'current_root_path': None}
        self.OptionList = ui.search_for_data_in_dir(self.config_dict['current_root_path'])
        self.opt_current_key = None
        self.opt = ttk.Combobox(self, values=self.OptionList, height=20, width=50)
        self.opt.place(x=640,y= 20, anchor='center')
        self.opt.bind("<<ComboboxSelected>>", self.ComboboxSelectedFunc)
        self.opt.pack()
        

        
            
    def choose_root(self):
        
        path = filedialog.askdirectory()
        self.config_buttons["test dir"]['button'].config(text=path)
#        self.buttons_dict["test dir"]['button'].pack()
        self.config_dict['current_root_path'] = path
        print(self.config_dict['current_root_path'])
        self.update()
        self.init_exploration()
        return 
#        self.buttons_dict["test dir"]['text'].set(path)
#        print(path)
#        self.set(path)
        
    def choose_modelpy(self):
        modelpy_path = filedialog.askopenfilename(initialdir = './',title = "Choose ai model.py", filetypes = (("py files","*.py"),("all files","*.*")))
        self.modelpy_path = modelpy_path
        self.eval = ui.Eval(self.modelpy_path)
        self.func_buttons_dict["Choose model button"]['name'] = self.modelpy_path
        
        if self.eval is None:
            self.eval = ui.Eval(self.modelpy_path)
        
        if not self.trainedpt_path is None:
            self.eval.load_Weights(self.trainedpt_path)
            
        self.func_buttons_dict["Choose model button"]['button'].config(text=modelpy_path[-20:])
        self.update()
        
        return 
    
    def choose_trainedpt(self):
        trainedpt_path = filedialog.askopenfilename(initialdir = './',title = "Choose ai trained.pt model ", filetypes = (("pt files","*.pt"),("all files","*.*")))
        self.trainedpt_path = trainedpt_path
        self.func_buttons_dict["Choose traind_model button"]['name'] = self.trainedpt_path
        if not self.modelpy_path is None:
            self.eval = ui.Eval(self.modelpy_path)
            
        if not self.eval is None:
            self.eval.load_Weights(trainedpt_path)
        
        self.func_buttons_dict["Choose traind_model button"]['button'].config(text=trainedpt_path[-20:])
        self.update()
        self.create_all_buttons()
        self.create_all_func_buttons()
        
        return 
    
    def run_ai_inference(self):
        print("run_inference")
#                    self.src_image, self.src_valid   
#            self.gt_image, self.gt_valid    
#            self.ai_image, self.ai_valid
        if not self.eval is None:
            if self.src_valid:
                temp = self.eval.from_image(self.src_image)
                temp = temp.astype("uint8")
                self.ai_image = cv2.cvtColor(temp, cv2.COLOR_GRAY2RGB)
                self.ai_valid = True
                self.create_all_buttons()
                self.create_all_func_buttons()
                
#                print (temp.shape)  
        return temp
#        print(modelpy_path)
        
    def show_image(self, img):
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
	    self.show_image(self.src_image)
            
    def show_src_with_gt(self):
	    img = np.where(self.colormap.apply(self.gt_image) > 0, 0, self.src_image)
	    self.show_image(img)
        
    def show_src_with_ai(self):
	    img = np.where(self.colormap.apply(self.ai_image) > 0, 0, self.src_image)
	    self.show_image(img)
        
    def show_gt_vs_ai(self):
	    img = self.src_image.copy()
	    img[self.gt_image > 0]= 0
	    img[self.ai_image > 0]= 255

	    self.show_image(img)
        
    def show_gt_histogram(self):
        hist, bins, hist_image = self.indexing_histogram.single_apply(self.gt_image[:,:,0])
        
        self.show_image(hist_image)
        
    def show_ai_histogram(self):
        hist, bins, hist_image = self.indexing_histogram.single_apply(self.ai_image[:,:,0])
        
        self.show_image(hist_image)
        
        
    def inAI_not_inGT(self):
        img = self.src_image.copy()
        gt_mask = self.gt_image == 0
        ai_mask = self.ai_image > 0
        img[np.logical_and(gt_mask, ai_mask)]= 0

        self.show_image(img)
        
    def inGT_not_inAI(self):
        img = self.src_image.copy()
        gt_mask = self.gt_image > 0
        ai_mask = self.ai_image == 0
        img[np.logical_and(gt_mask, ai_mask)]= 0

        self.show_image(img)
      
    def create_all_buttons(self):
        
        
        if self.src_valid:
            self.buttons_dict["Src  Img"] = {"func": self.show_src, 
                                             "name": "Src  Img",
                                             "show": True}
        else:
            if "Src  Img" in self.buttons_dict:
                self.buttons_dict["Src  Img"]["show"] = False
        
            
        
        if self.src_valid and self.gt_valid:
            self.buttons_dict["Src + GT"] =  {"func": self.show_src_with_gt, 
                                             "name": "Src + GT",
                                             "show": True}
        else:
            if "Src + GT" in self.buttons_dict:
                self.buttons_dict["Src + GT"]["show"] = False
            
        
        if self.src_valid and self.ai_valid:
            self.buttons_dict["Src + AI"] = {"func": self.show_src_with_ai, 
                                             "name": "Src + GT",
                                             "show": True}
        else:
            if "Src + AI" in self.buttons_dict:
                self.buttons_dict["Src + AI"]["show"] = False

            
        if self.gt_valid and self.ai_valid:
            self.buttons_dict["GT vs AI"] =  {"func": self.show_gt_vs_ai, 
                                             "name": "GT vs AI",
                                             "show": True}
            
            self.buttons_dict["inAI not inGT"] = {"func": self.inAI_not_inGT, 
                                                 "name": "inAI not inGT",
                                                 "show": True}
            
            self.buttons_dict["inGT not inAI"] = {"func": self.inGT_not_inAI, 
                                                 "name": "inAI not inGT",
                                                 "show": True}
        else:
            if "GT vs AI" in self.buttons_dict:
                self.buttons_dict["GT vs AI"]["show"]       = False
                self.buttons_dict["inAI not inGT"]["show"]  = False
                self.buttons_dict["inGT not inAI"]["show"]  = False
                
            
                                   
#        yt = 50
        nav_x_start = 0.15625
        nav_y_start = 0.03 *3
        nav_height=2
        nav_width = 10
        
        for name, function in self.buttons_dict.items():
#            tk.Button(self, text=name, command=function, height=nav_height, width=nav_width).place(relx=nav_x_start, rely= nav_y_start)

            
            if function['show']:
                self.buttons_dict[name]['button'] = tk.Button(self, text=function["name"], command=function["func"], height=nav_height, width=nav_width)
                self.buttons_dict[name]['button'].place(relx=nav_x_start, rely= nav_y_start)
                nav_x_start += 0.1015625
            else:
                if 'button' in self.buttons_dict[name]:
                    self.buttons_dict[name]['button'].destroy()
            
            
    def create_all_func_buttons(self):
                
        if not self.model_loaded:
            self.func_buttons_dict["Choose model button"] = {"func": self.choose_modelpy, 
                                                      "name": "Choose model.py file",
                                                                "show": True}
            
            self.func_buttons_dict["Choose traind_model button"] = {"func": self.choose_trainedpt,
                                                                     "name": "Choose traind_model_{*}.pt",
                                                                "show": True}
            self.model_loaded = True
            
                              
            
        if (not self.eval is None) and (not self.trainedpt_path is None):
            self.func_buttons_dict["run AI iference"] = {"func": self.run_ai_inference,
                                                          "name": "run AI iference",
                                                                "show": True}
            
        if self.gt_valid:
            self.func_buttons_dict["GT indexing histogram"] = {"func": self.show_gt_histogram, 
                                                                "name": "GT indexing histogram",
                                                                "show": True}
        else:
            if "GT indexing histogram" in self.func_buttons_dict:
                self.func_buttons_dict["GT indexing histogram"]["show"] = False
        
            
        if self.ai_valid:
            self.func_buttons_dict["AI indexing histogram"] = {"func": self.show_ai_histogram, 
                                                                "name": "AI indexing histogram",
                                                                "show": True}
        else:
            if "AI indexing histogram" in self.func_buttons_dict:
                self.func_buttons_dict["AI indexing histogram"]['show'] = False


            
#        if gt_valid and ai_valid:
#            func_buttons_dict["GT vs AI"] = self.show_gt_vs_ai
#            func_buttons_dict["inAI not inGT"] = self.inAI_not_inGT
#            func_buttons_dict["inGT not inAI"] = self.inGT_not_inAI                        
#        yt = 50
        nav_x_start = 0.8
        nav_y_start = 0.0625*3
        nav_height=2
        nav_width = 20
        
        for name, function in self.func_buttons_dict.items():
            if function['show']:
                self.func_buttons_dict[name]['button'] = tk.Button(self, text=function["name"], command=function["func"], height=nav_height, width=nav_width)
                self.func_buttons_dict[name]['button'].place(relx=nav_x_start, rely= nav_y_start)
            else:
                if 'button' in self.func_buttons_dict[name]:
                    self.func_buttons_dict[name]['button'].destroy()
                    
                
            nav_y_start += 0.05
            
        
    
    def ComboboxSelectedFunc(self, *args):
        temp = self.opt.get()
        if temp == self.opt_current_key:
            return
        else:  
            self.opt_current_key = self.opt.get()
            
        self.src_valid, self.gt_valid, self.ai_valid = False, False, False
		    
#		load_image( c_path + ".png"), load_image(c), load_mat_file_as_np(c_path[:-8] + name + "_skelSubP.mat")
        if self.opt_current_key in self.OptionList:
            key_name = self.opt_current_key.split(".")[0]
            self.key_full_path =  "{0}/{1}".format(self.config_dict['current_root_path'], key_name)
            

            self.src_image, self.src_valid   = ui.try_read_image("{0}.png".format(self.key_full_path))
            self.gt_image, self.gt_valid     = ui.try_read_image("{0}_indRef.png".format(self.key_full_path))
            self.ai_image, self.ai_valid     = ui.try_read_image("{0}_aiRef.png".format(self.key_full_path))
            
            self.create_all_buttons()
            
            self.create_all_func_buttons()
            
            
#            self.indexing_loss      = ui.read_image(self.config_dict['created_static_path'] + "idx_loss_all.png")
#            self.skeleton_loss    = ui.read_image(self.config_dict['created_static_path'] + "seg_loss_all.png")
#            self.segmentation_loss  = ui.read_image(self.config_dict['created_static_path'] + "skel_loss_all.png")
#            
            
            self.show_image(self.src_image)
#            self.show_graph(self.indexing_loss,     graph_name = 'indexing')
#            self.show_graph(self.skeleton_loss,     graph_name = 'skeleton')
#            self.show_graph(self.segmentation_loss, graph_name = 'segmentation')
            
            # show logs  
#            self.log_dict['current_csv_file'].config(text="current_csv_file: {0}".format(self.last_csv))
#            for log_key in self.csv_dict[self.c_key].keys():
#                if log_key in self.log_dict:
#                    self.log_dict[log_key].config(text="{0}: {1}".format(log_key, self.csv_dict[self.c_key][log_key]))
