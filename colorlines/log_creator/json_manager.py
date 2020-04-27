import json




socket_data = {
  'host_name': "",
  'port': 8000,
  'img_size': (4, 640, 640, 3),
  'skeleton_size': (4, 15000, 3),
  'start_str': "STA/",
  'end_str': "END/"
}

socket_data = {
  'host_name': "",
  'port': 8000,
  'img_size': (4, 640, 640, 3),
  'skeleton_size': (4, 15000, 3),
  'start_str': "STA/",
  'end_str': "END/"
}

model_config = {
                                              'model_size': 640,
                                              'last_trained_model_path': '/home/alon/dently_7_19/30_7_19/1.9.19/log/traind_model_165000.pt', #'/home/alon/dently_7_19/30_7_19/log/traind_model_27508.pt', # None - for starting new training 
                                              'log_folder_path': './log/',
                                              'use_adv': False,
                                              'indexing_max': 48,
                                              'indexing_min': 2,
                                              'use_adversial': True,
                                              'trained_advModel_path': None, 
                                              'training_max_steps': 500000,
                                              'training_batch_size': 2,
                                              'vis_training' : True,
                                              'save_log_after_N': 5000,
                                              'test_after_N': 5000,
                                              'test_steps': 100,
                                              'learning_rate': 1e-4,
                                              'training_data_path': [
                                                                     '/media/alon/SP PHD U3/AI_Ref/Rar/F__Records__ProtoTypeV2__Invitro__S_12.6.1__Tooth9_WhitePlast_Iron_WithBlood_181118/',
                                                                     '/media/alon/SP PHD U3/AI_Ref/Rar/F__Records__ProtoTypeV2__Invitro__S_12.6.2__Tooth10_WithoutBlood_z/',
                                                                     '/media/alon/SP PHD U3/AI_Ref/Rar/F__Records__ProtoTypeV2__Invitro__S_12.6.2__1-Tooth11_S_12.6.2_calibfile070319_20190310/',
                                                                     '/media/alon/SP PHD U3/AI_Ref/Rar/F__Records__ProtoTypeV2__Invitro__S_12.6.2__1-Tooth9_S_12.6.2_calibfile070319_20190310/',
                                                                     '/media/alon/SP PHD U3/AI_Ref/Rar/F__Records__ProtoTypeV2__Invitro__RecordsAdi_FastScan__1-Tooth9_S_12.6.2_CalibFile240319_z_20190325/',
                                                                     '/media/alon/SP PHD U3/AI_Ref/Rar/F__Records__ProtoTypeV2__Invitro__S_12.6.2__2-Tooth11_S_12.6.2_calibfile070319_20190310/',
                                                                     '/media/alon/SP PHD U3/AI_Ref/Rar/F__Records__ProtoTypeV2__Invitro__S_12.6.2__3-Tooth9_S_12.6.2_calibfile070319_20190310/'],
                                              'test_data_path': ['/media/alon/SP PHD U3/AI_Ref/Rar/F__Records__ProtoTypeV2__Invitro__S_12.6.2__3-Tooth9_S_12.6.2_calibfile070319_20190310/']
                                        
                            }
                               

def save_dict_as_json(tmp_dict, path = './config_data/model_config.json'):
    with open(path, 'w') as fp:
        json.dump(tmp_dict, fp, indent=2)
        
        
        
def read_dict_from_json(path = './config_data/data.json'):
    with open(path, 'r') as fp:
        data = json.load(fp)    
        
    return data


if __name__== "__main__":
    save_dict_as_json(model_config)
