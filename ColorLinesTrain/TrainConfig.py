import json
import os
import shutil


UI_config = {
  'HostName': "",
  'port': 8000,
  'InputShape':  None,
  'ReturnShape': (15000, 3),
  'BatchSize': 4,
  'start_str': "STA/",
  'end_str': "END/"
}



train_config = \
    {
     "TrainingLogDir": "./log",
     "modelsRootName": "trained_models",
     "ModelPyPath": '/media/nadav/HDD2T/CarrierSpace/sw/algo_nn_nadav_both/colorlines/nadav_index_model.py',
     
     "ModelConfig": { 
                        "ModelName": 'MyModel_Both_mod14_full_Test',
                        "InputShape": (640, 640, 3),
                        "UseIdx": True,
                        "UseSeg": False,
                        "UseSkel": False,
                },

      "RetrainModel": False,
      "RetrainModel_fullpath": '/media/nadav/HDD2T/CarrierSpace/sw/algo_nn_nadav_both/ColorLinesTrain/log/MyModel_Both_mod14_full_Test/trained_models/traind_model_78000.pt',
      "MaxSteps": 4000,
      "BatchSize": 20,
      "LogAfter": 100,
      "TestSteps": 40,
      "learningRate": 0.001, #0.0001
      
      "DatasetConfig":{    
          #"TrainingScansPath": ['/media/nadav/HDD2T/CarrierSpace/AI/ForAI_12.7.2_Accuracy_labelingTEC/Record.6062525a-de84-477e-8fc0-5eb8f340ce3c/AI_Labels_Shai_RT/',
          #                      '/media/nadav/HDD2T/CarrierSpace/AI/ForAI_12.7.2_Accuracy_labelingTEC/Record.5e9c1d5d-f16e-4cea-8ad7-32791f03d347/'],
          #"TestingScansPath": ['/media/nadav/HDD2T/CarrierSpace/AI/ForAI_12.7.2_Accuracy_labelingTEC/Record.6062525a-de84-477e-8fc0-5eb8f340ce3c/AI_Labels_Shai_RT/'],
          "TrainingScansPath": [
              '/media/nadav/HDD2T/CarrierSpace/AI/AI_Labels_inVivo_24/Record.0c54d637-0e4f-4f94-948d-9cf1931586a9_blueFilt_inVivoColorCube_GT_Extract/AI_Labels/',
#              '/media/nadav/HDD2T/CarrierSpace/AI/AI_Labels_inVivo_24/Record.1c8e5247-a5cc-4448-b630-b0bb23716395_blueFilt_inVivoColors_GT_Extract/AI_Labels/' # not good training set
#                                '/media/nadav/HDD2T/CarrierSpace/AI/AI_Labels_inVivo_24/Record.23500abf-6871-44b1-b4fb-db852ccf2fea_blueFilt_inVivoColor_extractGT/AI_Labels/'
                                ],
          "TestingScansPath": ['/media/nadav/HDD2T/CarrierSpace/AI/AI_Labels_inVivo_24/Record.23500abf-6871-44b1-b4fb-db852ccf2fea_blueFilt_inVivoColor_extractGT/AI_Labels/'],
          
          
          #"CreatedDemoAlgoCommit": "000000",
          #"Scan": "12.2.7",
          #"InputShape": (640, 640, 3),
          "UseIdx": True,
          "UseSeg": False,
          "UseSkel": True,
          
          "indexing_ext": "_indRef",
          "seg_ext": "_indSeg",
          "image_format": ".png"
          },
      
      "AugmantationConfig": {
              "Frequancy": 5, 
              "use_flip": True,
              "use_random_shifting": True,
              "use_down_up": False,
              "use_random_noise": True,
              "use_increase_brightness": True

              }
}    
     
socket_config = {
  'HostName': "",
  'port': 8000,
  'InputShape':  None,
  'ReturnShape': (15000, 3),
  'BatchSize': 1,
  'start_str': "STA/",
  'end_str': "END/"
}

model_config = {
    'ModelName': None,
    'ptFileName': None,
    'InputShape': None,
    'OutChannels': None, # 49 for indexing, 3 for seg
    'MinIdx': None,
    'MaxIdx': None           
}
                  

def save_dict_as_json(tmp_dict, path = './config/model_config.json'):
    with open(path, 'w') as fp:
        json.dump(tmp_dict, fp, indent=2)
        
        
        
def read_dict_from_json(path = './config_data/data.json'):
    with open(path, 'r') as fp:
        data = json.load(fp)    
        
    return data

def CreateIferenceDir():
    
    try:
        os.makedirs('{0}'.format(train_config["TrainingLogDir"]))
    except FileExistsError:
        print("TrainingLogDir directory already exists")
        pass
    
    try:
        os.makedirs('{0}/{1}'.format(train_config["TrainingLogDir"], train_config["ModelConfig"]["ModelName"]))
    except FileExistsError:
        print("ModelName directory already exists")
        pass
    
    try:
        os.makedirs('{0}/{1}/{2}'.format(train_config["TrainingLogDir"], train_config["ModelConfig"]["ModelName"], train_config["modelsRootName"]))
    except FileExistsError:
        print("ModelName directory already exists")
        pass
    
    socket_config_out = socket_config
    socket_config_out["InputShape"] =  train_config["ModelConfig"]["InputShape"]
    
    try:
        save_dict_as_json(socket_config_out, '{0}/{1}/socket_config.json'.format(train_config["TrainingLogDir"], train_config["ModelConfig"]["ModelName"]))
        print("socket_config created")
    except:
        print("can't create socket_config.json")
        pass
    
    for key in model_config:
        if key in train_config['ModelConfig']:
            model_config[key] = train_config['ModelConfig'][key]
            
    try:
        save_dict_as_json(model_config, '{0}/{1}/model_config.json'.format(train_config["TrainingLogDir"], train_config["ModelConfig"]["ModelName"]))
        print("model_config created")
    except:
        print("can't create model_config.json")
        pass
            
    
    if not os.path.isfile(train_config["ModelPyPath"]):
        print("{0} not exist".format(train_config["ModelPyPath"]))
    else:
        shutil.copy(train_config["ModelPyPath"], '{0}/{1}/model.py'.format(train_config["TrainingLogDir"], train_config["ModelConfig"]["ModelName"]))
        print("model found")
    

#if __name__== "__main__":
#    CreateIferenceDir()
    

#