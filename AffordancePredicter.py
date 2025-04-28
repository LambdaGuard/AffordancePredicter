import os
from gorilla.config import Config
from os.path import join as opj
from .utils import *
import torch
import open3d as o3d
import numpy as np
import trimesh
import matplotlib.pyplot as plt

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc, centroid, m

class AffordancePredicter():
    def __init__(self, config_path, checkpoint_path):
        self.cfg = Config.fromfile(config_path)
        self.model = build_model(self.cfg).cuda()
        
        if checkpoint_path == None:
            print("Please specify the path to the saved model")
            exit()
        else:
            print("Loading affordance model....")
            print("Checkpoint path: ", checkpoint_path)
            _, exten = os.path.splitext(checkpoint_path)
            if exten == '.t7':
                self.model.load_state_dict(torch.load(checkpoint_path))
            elif exten == '.pth':
                check = torch.load(checkpoint_path)
                self.model.load_state_dict(check['model_state_dict'])
            print("Affordance Model loaded successfully!")
        self.model.eval()
        self.affordance = self.cfg.training_cfg.val_affordance
        print("Affordance categories: ", self.affordance)

        # run a dummy forward pass to initialize the model
        dummy_data = torch.randn(1, 3, 2048).cuda()
        with torch.no_grad():
            _ = self.model(dummy_data, self.affordance)
    
    def predict(self, input_pc):
        '''
        input_pc: [N,3] numpy array
        Return:
        pred: [N,] numpy array
        '''
        datas, centroid, m = pc_normalize(input_pc)
        datas = torch.from_numpy(datas).unsqueeze(0).cuda()
        datas = datas.permute(0, 2, 1).contiguous().float()
        with torch.no_grad():
            afford_pred = self.model(datas, self.affordance)
        afford_pred = afford_pred.permute(0, 2, 1).cpu().numpy()
        afford_pred = np.argmax(afford_pred, axis=2)[0]
        return afford_pred
