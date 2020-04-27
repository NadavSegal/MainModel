import tkinter as tk
from TrainConfig import train_config
import os
import csv

from UI_pages.TrainPage import Page as TrainPage
from UI_pages.ExplorationPage import Page as ExplorationPage


LARGE_FONT= ("Verdana", 12)

class SeaofBTCapp(tk.Tk):

    def __init__(self, UI_config, *args, **kwargs):
        self.UI_config = UI_config
        tk.Tk.__init__(self, *args, **kwargs)
        container = tk.Frame(self)

        container.pack(side="top", fill="both", expand = True)

        container.grid_rowconfigure(0, weight=100)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        for F in (MainMenu, TrainPage, ExplorationPage):

            frame = F(container, self)

            self.frames[F] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(MainMenu)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()
        
    def show_mainmenu(self):
        frame = self.frames[MainMenu]
        frame.tkraise()

        
class MainMenu(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self,parent)
#        label = tk.Label(self, text="Start Page", font=LARGE_FONT)
#        label.pack(pady=10,padx=10)
#        Tkinter.Button(self, text="Src Img", command=self.show_src, height=2, width=10)
        button = tk.Button(self, 
                           text="Color Line Train",
                           height=50, 
                           width=25,
                           command=lambda: controller.show_frame(TrainPage))
        
        button.pack()
        button.place(x=320,y= 0)

        button2 = tk.Button(self, 
                            text="Color Line Data Exploration",
                            height=50, 
                            width=25,
                            command=lambda: controller.show_frame(ExplorationPage))
        button2.pack()
        button2.place(x=760,y= 0)

       
def find_last_csv(path_to_search):
    if not os.path.isdir(path_to_search):
        return None
    
    last_csv = None
    max_idx = 0
    for path in os.listdir(path_to_search):
        if "csv" in path:
            number_str = path.split(".csv")[0].split("_")[1]
            if int(number_str) > max_idx:
                max_idx = int(number_str) 
                last_csv = path
                
    return last_csv

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


static_folder_path = "{0}/{1}/static/".format(train_config['TrainingLogDir'], train_config['ModelConfig']['ModelName'])
UI_config = {
                    "created_static_path": static_folder_path, 
                    "last_created_csv_file": find_last_csv(static_folder_path)
                 }


app = SeaofBTCapp(UI_config)
app.geometry("1280x800")
app.mainloop()