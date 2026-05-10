import os
import json
import random
import shutil
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility
from pyquaternion import Quaternion
import numpy as np

def extract_nuscenes_samples(dataroot='/home/curryflink/下载/v1.0-mini', version='v1.0-mini', n_samples=5):
    """提取 nuScenes 图片及 3D->2D 投影真值"""
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
    
    # 选定几个具有代表性的场景描述关键词
    good_scenarios = ['night', 'rain', 'intersection', 'construction', 'bus']
    selected_scenes = []
    
    # 尝试寻找符合关键词的场景
    for scene in nusc.scene:
        if any(word in scene['description'].lower() for word in good_scenarios):
            selected_scenes.append(scene)
        if len(selected_scenes) >= n_samples:
            break
            
    # 如果关键词没找够，补齐
    if len(selected_scenes) < n_samples:
        for scene in nusc.scene:
            if scene not in selected_scenes:
                selected_scenes.append(scene)
            if len(selected_scenes) >= n_samples:
                break

    extracted_data = []
    
    for i, scene in enumerate(selected_scenes):
        sample_token = scene['first_sample_token']
        sample = nusc.get('sample', sample_token)
        
        camera_channel = 'CAM_FRONT'
        sd_record = nusc.get('sample_data', sample['data'][camera_channel])
        
        # 1. 复制图片
        src_path = os.path.join(dataroot, sd_record['filename'])
        dst_name = f"nusc_test_{i+1}.png"
        shutil.copy(src_path, dst_name)
        
        # 2. 获取投影真值 (Ground Truth)
        # 获取相机内参和外参
        cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
        sensor_intrinsic = np.array(cs_record['camera_intrinsic'])
        
        # 获取真值框并在当前视角下投影
        _, boxes, camera_intrinsic = nusc.get_sample_data(sd_record['token'], box_vis_level=BoxVisibility.ANY)
        
        gt_annotations = []
        for box in boxes:
            # 只要车辆类别
            if 'vehicle' not in box.name:
                continue
                
            # 将 3D 框投影到 2D 像素平面
            # corners 是 3x8 的矩阵
            corners = box.corners()
            # 投影到像素坐标 (view_points 函数处理内参投影)
            pts = view_points(corners, sensor_intrinsic, normalize=True)[:2, :]
            
            gt_annotations.append({
                "category": box.name,
                "corners_2d": pts.T.tolist(), # 转为 8x2 列表
                "center_dist": np.linalg.norm(box.center) # 距离相机的距离
            })

        extracted_data.append({
            "id": i + 1,
            "filename": dst_name,
            "scene_description": scene['description'],
            "gt_annotations": gt_annotations,
            "camera_intrinsic": sensor_intrinsic.tolist()
        })
        print(f"已提取高质量场景 {i+1}: {scene['description']} (包含 {len(gt_annotations)} 个车辆真值)")

    with open("nusc_meta.json", "w", encoding="utf-8") as f:
        json.dump(extracted_data, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    extract_nuscenes_samples()
