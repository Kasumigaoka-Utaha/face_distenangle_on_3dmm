import torch
import cv2
import numpy as np
import os
import json
import glob

def load_dir(path, start, end):
    lmss = []
    for i in range(start, end):
        if os.path.isfile(os.path.join(path, str(i) + '.lms')):
            lms = np.loadtxt(os.path.join(
                path, str(i) + '.lms'), dtype=np.float32)
            lmss.append(lms)
    lmss = np.stack(lmss)
    lmss = torch.as_tensor(lmss).cuda()
    return lmss

def load_id(path):
    with open(path,'r') as static:
        json_data = json.load(static)
        id = json_data['id']
        focal = json_data['focal']
    return id,focal

def load_exp(path):
    exp_list = []
    euler_list = []
    trans_list = []
    for file in glob.glob(path+'*.json'):
        name = os.path.basename(file)
        if name == 'static_params.json':
            continue
        with open(file,'r') as data:
            data_info = json.load(data)
            exp_temp = data_info['exp']
            exp_para = [(x-min(exp_temp))/(max(exp_temp)-min(exp_temp)) for x in exp_temp]
            euler_para = data_info['euler']
            trans_para = data_info['trans']
        exp_list.append(exp_para)
        euler_list.append(euler_para)
        trans_list.append(trans_para)
    return exp_list,euler_list,trans_list
