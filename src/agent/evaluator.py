import numpy as np
from typing import List, Tuple, Dict, Optional
from src.utils.schema import Annotation

def calc_box_ratio_err(keypoints: List[Tuple[float, float]]) -> float:
    """
    计算3D Box的几何合理性代理：长宽比偏差。
    理想轿车比例约为 1.5:1, 卡车 2.5:1。
    这里简化处理，计算投影后的包围盒比例偏差。
    """
    if not keypoints or len(keypoints) < 8:
        return 1.0
    
    pts = np.array(keypoints)
    min_pts = np.min(pts, axis=0)
    max_pts = np.max(pts, axis=0)
    width = max_pts[0] - min_pts[0]
    height = max_pts[1] - min_pts[1]
    
    if height == 0: return 1.0
    ratio = width / height
    
    # 假设目标是轿车 (1.5)
    target_ratio = 1.5
    return abs(ratio - target_ratio) / target_ratio

def calc_lane_parallelism(lane_coeffs: List[List[float]]) -> float:
    """
    计算车道线平行度（曲率差）。
    对于 y=ax²+bx+c，a项决定曲率。
    """
    if len(lane_coeffs) < 2:
        return 0.0 # 无法计算平行度，暂不扣分
    
    a_values = [coeffs[0] for coeffs in lane_coeffs]
    # 计算a项的标准差，越小越平行
    return np.std(a_values)

def evaluate_geometry(pred: Annotation, pseudo_gt: Optional[Annotation] = None) -> Dict[str, float]:
    """
    模块 4：几何指标审核。
    返回量化指标，禁止文本描述。
    """
    scores = {}
    
    # 1. 3D Box 几何合理性
    box_err = calc_box_ratio_err(pred.bbox_3d_keypoints)
    scores['box_integrity'] = max(0.0, 1.0 - box_err)
    
    # 2. 车道线平行度
    lane_err = calc_lane_parallelism(pred.lane_coeffs)
    scores['lane_consistency'] = max(0.0, 1.0 - lane_err * 100) # 放大误差以便观察
    
    # 3. 像素级一致性 (如果存在参考/上一轮结果)
    if pseudo_gt and pseudo_gt.bbox_3d_keypoints:
        pts_pred = np.array(pred.bbox_3d_keypoints)
        pts_gt = np.array(pseudo_gt.bbox_3d_keypoints)
        # 简化版 PCK
        dist = np.linalg.norm(pts_pred - pts_gt, axis=1)
        scores['pck'] = np.mean(dist < 5.0) # 5像素阈值
    else:
        scores['pck'] = 1.0 # 无参考时默认满分
        
    # 4. 综合得分
    scores['total'] = 0.4 * scores['box_integrity'] + 0.4 * scores['lane_consistency'] + 0.2 * scores['pck']
    
    return scores
