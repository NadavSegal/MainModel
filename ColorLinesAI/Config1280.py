import json
import os

socket_config = {
  'host_name': "",
  'port': 8000,
  'DevId': "NONE",
  'InputShape': (800, 1280, 3),
  'ReturnShape': (15000, 8),
  'startMark': "STA/",
  'endMark': "END/"
}

model_config = {
    'ModelName': '1280_fast_6conv_312_Top',
    'ptFileName': 'traind_model_118951' + '.pt',
    'InputShape': (800, 1280, 3),
    'OutChannels': 49 + 3, # 49 for indexing, 3 for seg
    'MaxIdx': 48,
    'MinIdx': 2           
}
        
AI_config = {
    'Mode': None,
    'ModelsPath': './models',
    'VisMode': 'Inference',
    'BatchSize': 1,
    'ModelConfig': model_config,
    'SocketConfig': socket_config,
    'LogDir': './log/'
 
}       

def ConfigTester():
    temp_config = AI_config
    model_input = temp_config['ModelConfig']['InputShape'] 
    socket_input = temp_config['SocketConfig']['InputShape'] 
    model_path = temp_config['ModelsPath']
    model_name = temp_config['ModelConfig']['ModelName']
    pt_name = temp_config['ModelConfig']['ptFileName']
   
    assert (model_input == socket_input), \
           "ModelConfig image shape: {0} != SocketConfig image shape: {1}".format(model_input, socket_input)
           
    assert (not os.path.isfile("{0}/{1}/model.py".format(model_path, model_name)) == False), \
        "Choosen Model {0}/{1}/model.py dosen't exist ".format(model_path, model_name)
        
    assert (not os.path.isfile("{0}/{1}/{2}".format(model_path, model_name, pt_name)) == False), \
        "Choosen ptFileName {0}/{1}/{2} dosen't exist ".format(model_path, model_name, pt_name)
        
    return True
                  

def save_dict_as_json(tmp_dict, path = './config/model_config.json'):
    with open(path, 'w') as fp:
        json.dump(tmp_dict, fp, indent=2)
        
        
        
def read_dict_from_json(path = './config_data/data.json'):
    with open(path, 'r') as fp:
        data = json.load(fp)    
        
    return data


if __name__== "__main__":
    ConfigTester()
