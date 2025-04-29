import os
from os.path import join as opj
import torch
import open3d as o3d
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from .AffordancePredicter import AffordancePredicter
import matplotlib.patches as mpatches

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc, centroid, m

def run():
    # config_path = 'affordance_detection/config/openad_pn2/full_shape_cfg.py'
    # checkpoint_path = 'affordance_detection/checkpoint/best_model_openad_pn2_estimation.t7'
    config_path = 'affordance_detection/config/openad_pn2/partial_view_cfg.py'
    checkpoint_path = 'affordance_detection/checkpoint/best_model_openad_pn2_partial.t7'
    predicter = AffordancePredicter(config_path, checkpoint_path)

    # input_path = 'affordance_detection/test_data/dense_points.ply'
    input_path = 'affordance_detection/test_data/a bottle drink_completed.ply'
    # input_path = 'test_data/2_object_lunch bag_comp.ply'
    # input_path = 'affordance_detection/test_data/1_object_remote.ply'
    # input_path = 'test_data/gt.xyz'
    # input_path = 'affordance_detection/test_data/partial.ply'
    if input_path.endswith('.ply'):
        input_mesh = o3d.io.read_point_cloud(input_path)
        input_pc = np.array(input_mesh.points)
    elif input_path.endswith('.xyz'):
        input_pc = np.loadtxt(input_path)

    with torch.no_grad():
        afford_pred = predicter.predict(input_pc)


    val_affordance = predicter.affordance
    # 创建一个包含与 affordance 类别数相同颜色的 colormap
    cmap = plt.get_cmap('tab20', len(val_affordance))
    # 将每个预测标签映射到 RGB 颜色，结果为 (8192, 3) 的 numpy.ndarray
    afford_pred_color = cmap(afford_pred)[:, :3]

    # 生成每个类别对应的颜色
    colors = cmap(np.arange(len(val_affordance)))[:, :3]

    # 为每个 affordance 创建一个 patch，用于图例
    handles = [
        mpatches.Patch(color=colors[i], label=val_affordance[i])
        for i in range(len(val_affordance))
    ]

    plt.figure(figsize=(1,1))
    plt.legend(handles=handles, ncol=2, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.axis('off')
    plt.show()
    plt.savefig('affordance_detection/test_data/legend.png', bbox_inches='tight', dpi=300)

    output_mesh = o3d.geometry.PointCloud()
    output_mesh.points = o3d.utility.Vector3dVector(input_pc)
    output_mesh.colors = o3d.utility.Vector3dVector(afford_pred_color)
    o3d.io.write_point_cloud('affordance_detection/test_data/affordance_color.ply', output_mesh)
