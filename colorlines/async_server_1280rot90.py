from __future__ import print_function
from __future__ import print_function
import sys
import socket
import numpy as np
import time
import json_manager
import os

#from dently_model import Eval
from Dently_eval import Eval

import json, os
import sys, getopt
import IpcServerLib_ext
import array
mode='640'




class eval_socket:
    def __init__(self, dict_path="./config_data/data_1280rot90.json", mode='1280rot90'):
        self.isSlow = False #slow=Alon, fast=Marina

        if self.isSlow:
            self.HOST = bytes(self.dict['host_name'], "utf-8")  # Standard loopback interface address (localhost)
            self.PORT = self.dict['port']  # Port to listen on (non-privileged ports are > 1023)
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.bind((self.HOST, self.PORT))
        else:
            self.DevId = "NONE"

        self.dict = json_manager.read_dict_from_json(dict_path)
        self.model = Eval(mode='1280')
        self.model.load_Weights(self.dict['model'])

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
        if self.isSlow:
            return self.listen_and_eval_slow()
        else:
            return self.listen_and_eval_fast()

    def listen_and_eval_slow(self): #slow/alon
        self.socket.listen(1)
        start_str = bytes(self.dict['start_str'], "utf-8")
        end_str = bytes(self.dict['end_str'], "utf-8")
        print("listening (slow)")
        data = b''
        FRAME_LENGTH = 800 * 1280 * 3
        c, a = self.socket.accept()
        while True:

            data = b''
            st_time = time.time()

            while len(data) < FRAME_LENGTH:
                block = c.recv(FRAME_LENGTH)
                print("Recv {0}".format(len(block)))
                if not block: break
                data += block

            unserialized_input = np.frombuffer(data, dtype=np.uint8).reshape(800, 1280, 3)
            st = time.time()
            real_image, skeleton, indexing_output, indexing_argmax, seg_output, seg_argmax, skeleton_aslist = self.model.from_img(
                unserialized_input)
            print(skeleton_aslist.shape)
            # st = time.time()
            print(time.time() - st)

            model_output = skeleton_aslist
            serialized_data = model_output.tobytes()
            temp_msg = start_str + serialized_data + end_str
            if(0):
                c.sendall(temp_msg)
            else:
                self.sendData(temp_msg)

            print(unserialized_input.shape)
            print(time.time() - st_time)

        c.close()


    def listen_and_eval_fast(self): #fast/marina
        start_str = bytes(self.dict['start_str'], "utf-8")
        end_str = bytes(self.dict['end_str'], "utf-8")
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

            unserialized_input = np.frombuffer(data, dtype=np.uint8).reshape(800, 1280, 3)
            st = time.time()
            real_image, skeleton, indexing_output, indexing_argmax, seg_output, seg_argmax, skeleton_aslist = self.model.from_img_for_socket(
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
mainServer = eval_socket(dict_path="./config_data/data_1280rot90.json", mode='1280rot90')
mainServer.listen_and_eval()






