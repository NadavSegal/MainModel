#import Config1280 as Config
import Config640 as Config
import torch
import importlib.util
import numpy as np
import os 
import json
import time
import IpcServerLib_ext
from cv2 import imwrite, cvtColor, COLOR_BGR2RGB

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

class AImodel:
    def __init__(self):
        # Test config params before init 
        Config.ConfigTester()
        self.config = Config.AI_config
        self.config['GPU'] = torch.cuda.is_available()
        self.model = self.importModel()
        self.loadWeights()
        
    def importModel(self):
        model_path = self.config['ModelsPath']
        model_name = self.config['ModelConfig']['ModelName']
        spec = importlib.util.spec_from_file_location("DentlyNet", "{0}/{1}/model.py".format(model_path, model_name))
        foo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(foo)
        return foo.DentlyNet()
               
    def loadWeights(self):
        pt_full_path = "{0}/{1}/{2}".format(self.config['ModelsPath'], \
                                            self.config['ModelConfig']['ModelName'], 
                                            self.config['ModelConfig']['ptFileName'])
        print("loading trained model from path: {0}".format(pt_full_path))
        
        model_dict = self.model.state_dict()
        if self.config['GPU']:
            last_state = torch.load(pt_full_path) 
        else:
            last_state = torch.load(pt_full_path, map_location={'cuda:0': 'cpu'})
            
        new_state = {k: v for k, v in last_state.items() \
                     if k in model_dict.keys()} # 1. filter out unnecessary keys        
        model_dict.update(new_state) # 2. overwrite entries in the existing state dict
        self.model.load_state_dict(model_dict)
        
            
        
class AIeval:
    def __init__(self):        
        self.ai = AImodel()
        self.ai.model.eval()
        self.frame_counter = 0
        
        if self.ai.config['GPU']:
            self.ai.model.cuda()
            self.dtype = torch.cuda.FloatTensor
            self.dtype_l = torch.cuda.LongTensor
        else:
            self.dtype = torch.FloatTensor
            self.dtype_l = torch.LongTensor
    
    def preprocess(self, x=None):
        x = torch.from_numpy(x).type(self.dtype)
        if self.ai.config['GPU']:
            x = x.cuda()
            
        x = x.permute(2,0,1)
        x = torch.div(x , 255.0)
        x = torch.unsqueeze(x, 0)
            
        return x

    def getNormCord(self, idxs):
        o_c = 4
        idxs_uint8 = np.zeros((np.shape(idxs)[0], o_c), 'uint8')
        idxs_uint8[:, 0] = idxs[:, 0] / 256
        idxs_uint8[:, 1] = idxs[:, 0] % 256
        idxs_uint8[:, 2] = idxs[:, 1] / 256
        idxs_uint8[:, 3] = idxs[:, 1] % 256

        return idxs_uint8

    def from_img(self, img):
        if self.ai.config['SocketConfig']['InputShape'] == (1280, 800, 3) and \
            img.shape == (800,1280,3):
            img = np.rot90(img)
        
        assert img.shape == self.ai.config['SocketConfig']['InputShape'],\
           "SocketConfig image shape: {0} != input img shape: {1}" .format(
               self.ai.config['SocketConfig']['InputShape'], img.shape)

        
        x = self.preprocess(img)

        #indexing_output, indexing_argmax, seg_output, seg_argmax, _ = self.ai.model.forward_eval(x)
        #indexing_output, indexing_argmax = self._process_indexing(indexing_output, indexing_argmax)

        #skeleton = self._process_skelaton(indexing_output, indexing_argmax)
        #_, skeleton, skeleton_stat = self.ai.model.forward(x)
        final_res, skeleton_final, skeleton_pred, indexing_tensor, indexing_argmax, indexing_pred = self.ai.model.forward_eval(x)

        skeleton_final = torch.squeeze(skeleton_final)
        indexing_pred = torch.squeeze(indexing_pred)
        final_res = 2*torch.squeeze(final_res )
        #final_res[indexing_pred < 0.9] = 0 #0.9 / torch.mean(indexing_pred[skeleton_final>0]) = 0.9794
        #final_res[skeleton_final < 1] = 0 #1, 2.735 / torch.mean(skeleton_final[skeleton_final>0]) = 2.7083

        self.frame_counter += 1
        #seg_argmax = torch.squeeze(seg_argmax, 0)

        idxs = final_res.nonzero()
        x = idxs[:, 0]
        y = idxs[:, 1]
        values = final_res[x, y]
        values = torch.unsqueeze(values, -1)

        #seg_values = seg_argmax[x, y]
        #seg_values = torch.unsqueeze(seg_values, -1)
        
        with torch.no_grad():
            if self.ai.config['GPU']:
                torch.cuda.synchronize()
                #seg_values = seg_values.cpu().data.numpy()
                values = values.cpu().data.numpy()
                idxs = idxs.cpu().data.numpy()

        idxs_uint8 = self.getNormCord(idxs)
        #subpixel_values = np.zeros_like(seg_values)
        #subtype_values = np.zeros_like(seg_values)
        output_aslist = np.concatenate((idxs_uint8, values), 1)

        pad = self.ai.config['SocketConfig']['ReturnShape'][0] - \
              idxs_uint8.shape[0]

        if pad > 0:
            o_c = self.ai.config['SocketConfig']['ReturnShape'][1]
            #pad_with_zeros = np.zeros((pad, o_c),'uint8')
            pad_with_zeros = np.zeros((pad, 5), 'uint8')
            output_aslist = np.concatenate((output_aslist, pad_with_zeros), 0)
            pad_with_zeros = np.zeros((15000, 3), 'uint8')
            output_aslist = np.concatenate((output_aslist, pad_with_zeros), 1)

        else:
            output_aslist = output_aslist[:self.ai.config['SocketConfig']['ReturnShape'][0],:]

        assert output_aslist.shape == self.ai.config['SocketConfig']['ReturnShape'], \
            "SocketConfig ReturnShape shape: {0} != output_aslist shape: {1}" .format(
               self.ai.config['SocketConfig']['ReturnShape'], output_aslist.shape)

        return output_aslist.astype('uint8')
        #return img, skeleton, indexing_output, indexing_argmax, seg_output, seg_argmax, skeleton_aslist.astype('uint8')
        
    
    def _process_indexing(self,
                          indexing_output, 
                          indexing_argmax,
                          th=0.5 ):
        
        value_max = indexing_output.type(self.dtype)
        value_max = torch.where((value_max >= th), value_max, torch.FloatTensor(value_max.size(0),  value_max.size(1), value_max.size(2)).zero_().type(self.dtype))
        indexing_output = torch.where((value_max > 0), indexing_output.type(self.dtype), torch.FloatTensor(value_max.size(0),  value_max.size(1), value_max.size(2)).zero_().type(self.dtype))
        indexing_argmax = torch.where((value_max > 0), indexing_argmax.type(self.dtype), torch.FloatTensor(value_max.size(0),  value_max.size(1), value_max.size(2)).zero_().type(self.dtype))
        
        return indexing_output, indexing_argmax
    
    def _process_seg(self,
                          seg_output, 
                          seg_argmax,
                          th=0.6):
        
        
        value_max = seg_output.type(self.dtype)
        value_max = torch.where((value_max >= th), value_max, torch.FloatTensor(value_max.size(0),  value_max.size(1), value_max.size(2)).zero_().type(self.dtype))
        seg_output = torch.where((value_max > 0), seg_output.type(self.dtype), torch.FloatTensor(value_max.size(0),  value_max.size(1), value_max.size(2)).zero_().type(self.dtype))
        seg_argmax = torch.where((value_max > 0), seg_argmax.type(self.dtype), torch.FloatTensor(value_max.size(0),  value_max.size(1), value_max.size(2)).zero_().type(self.dtype))
        
        return seg_output, seg_argmax
    
    
    def _process_skelaton(self, process_indexing, indexing_argmax):
        

        test11 = self.ai.model.mp2d(process_indexing.type(self.dtype))
        process_indexing = torch.where((indexing_argmax != 0), process_indexing.type(self.dtype), torch.FloatTensor(process_indexing.size(0),  process_indexing.size(1), process_indexing.size(2)).zero_().type(self.dtype))

        indexing_output = torch.where((process_indexing == test11), indexing_argmax.type(self.dtype), torch.FloatTensor(process_indexing.size(0),  process_indexing.size(1), process_indexing.size(2)).zero_().type(self.dtype))

        return indexing_output
    
class eval_socket:
    def __init__(self):
        self.DevId = "NONE"
        self.frame_counter = 0
#        self.dict = json_manager.read_dict_from_json(dict_path)
        self.model = AIeval()
#        self.model.load_Weights(self.dict['model'])

    def parseMeta(self, joMeta): #from marina
        DevId = ""
        Status = ""
        if ('Header' in joMeta):
            RxHeader = joMeta['Header']
            if ('DevId' in RxHeader):
                DevId = RxHeader['DevId']
        if ('Status' in joMeta):
            Status = joMeta['Status']

        self.DevId = DevId
        return Status

    def sendData(self, bytesData): #from Marina
        AlgoHeader = {}
        AlgoHeader['TestName'] = 'aaa'
        jstr = json.dumps(AlgoHeader)
        AlgoHeaderArg = bytes(jstr, encoding='utf8')
        #bytesData = bytes("123", 'utf-8')

        Header = {}
        Header['IpcPktDataSzBytes'] = len(bytesData)
        Header['DevId'] = self.DevId
        jstr = json.dumps(Header)
        HeaderArg = bytes(jstr, encoding='utf8')

        #print(HeaderArg)
        #print(AlgoHeaderArg)
        #print("call c")
        IpcServerLib_ext.SendPacket(HeaderArg, AlgoHeaderArg, bytesData)
        return

    def listen_and_eval(self):
        return self.listen_and_eval_fast()


    def listen_and_eval_fast(self): #fast/marina
        print("listening (fast)")
        while True:
            data = b'';
            try:
                [meta,data] = IpcServerLib_ext.GetPacket()
                joMeta = json.loads(meta)
                Status = self.parseMeta(joMeta)
                #print(Status)
                #You should DevId that you received when you send a response
            except Exception as e:
                print(e)
            #print("Get after")

            unserialized_input = np.frombuffer(data, dtype=np.uint8).reshape(self.model.ai.config['SocketConfig']['InputShape'])
            unserialized_input = cvtColor(unserialized_input, COLOR_BGR2RGB)
            #imwrite("C:/sw/algo_nn_alon/testlog/{0}.jpg".format(self.frame_counter), unserialized_input)
            self.frame_counter += 1
            #cv2.waitKey(10)
            
            st = time.time()
            skeleton_aslist = self.model.from_img(
                unserialized_input)
            print(skeleton_aslist.shape)
            # st = time.time()
            #print(time.time() - st)

            model_output = skeleton_aslist
            serialized_data = model_output.tobytes()
            self.sendData(serialized_data)




if os.getenv('VIRTUAL_ENV'):
    print('Using Virtualenv')
else:
    print('Not using Virtualenv')
mainServer = eval_socket()
mainServer.listen_and_eval()
#    def _draw_results(self,
#                      src_image,
#                      mask):
#        
#        b = mask[..., None] > 0
#        bbb = np.concatenate([b, b, b], axis=2)
#        
#        src_image[bbb > 0] = 0
#        
#        return src_image   

#b = AIeval()
#import cv2
#img = cv2.imread('/Users/alontetro/Downloads/00003594.png')
#real_image, skeleton_o, indexing_output, indexing_argmax, seg_output, seg_argmax, skeleton_aslist = b.from_img(img)
#temp = b._draw_results(real_image, skeleton_o)
#cv2.imshow("pred_indexing",  (temp).astype("uint8"))
#cv2.waitKey(0)
#b.model.eval()
#a, d,e,k, i = b.model.forward_eval()