import os
import sys
# from gorilla.config import Config
from third_party.AffordancePredicter.config.config import Config
from os.path import join as opj
from .utils import *
import torch
# import open3d as o3d
import numpy as np
# import trimesh
# import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN


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
        pcd = input_pc
        eps_value = 0.01          # Maximum distance between two samples for one to be considered as in the neighborhood of the other.
        min_samples_value = 10   # Minimum number of points required to form a dense region.

        # Run DBSCAN
        db = DBSCAN(eps=eps_value, min_samples=min_samples_value)
        labels = db.fit_predict(pcd)

        valid_labels = labels[labels != -1]
        cluster_counts = np.bincount(valid_labels)
        largest_cluster_label = np.argmax(cluster_counts)

        # Get the indices of the points in the largest cluster
        largest_cluster_indices = np.where(labels == largest_cluster_label)[0]
        input_pc = pcd[largest_cluster_indices]

        datas, centroid, m = pc_normalize(input_pc)
        datas = torch.from_numpy(datas).unsqueeze(0).cuda()
        datas = datas.permute(0, 2, 1).contiguous().float()
        with torch.no_grad():
            afford_pred = self.model(datas, self.affordance)
        afford_pred = afford_pred.permute(0, 2, 1).cpu().numpy()
        afford_pred = np.argmax(afford_pred, axis=2)[0]
        vis = True
        if vis:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            import open3d as o3d
            val_affordance = self.affordance
            cmap = plt.get_cmap('tab20', len(val_affordance))
            # 将每个预测标签映射到 RGB 颜色，结果为 (8192, 3) 的 numpy.ndarray
            afford_pred_color = cmap(afford_pred)[:, :3]

            # 生成每个类别对应的颜色
            colors = cmap(np.arange(len(val_affordance)))[:, :3]

            # 为每个 affordance 创建一个 patch，用于图例
            
            # handles = [
            #     mpatches.Patch(color=colors[i], label=val_affordance[i])
            #     for i in range(2)
            # ]

            # plt.figure(figsize=(1,1))
            # plt.legend(handles=handles, ncol=2, bbox_to_anchor=(1.05, 1), loc='upper left')
            # plt.axis('off')
            # plt.show()
            # plt.savefig('legend.png', bbox_inches='tight', dpi=300)

            output_mesh = o3d.geometry.PointCloud()
            output_mesh.points = o3d.utility.Vector3dVector(input_pc)
            output_mesh.colors = o3d.utility.Vector3dVector(afford_pred_color)
            o3d.io.write_point_cloud('affordance_color.ply', output_mesh)
            o3d.visualization.draw_geometries([output_mesh])
        return afford_pred
