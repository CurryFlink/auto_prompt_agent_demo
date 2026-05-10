from typing import List, Tuple, Dict, Optional
import numpy as np

# --- 模块 0：前置约定 ---

class Annotation:
    def __init__(self, 
                 bbox_3d_keypoints: List[Tuple[float, float]] = None, 
                 lane_coeffs: List[List[float]] = None,
                 cam_params: Dict = None,
                 meta: Dict = None):
        # 8个角点像素坐标 [(x1,y1), ...]
        self.bbox_3d_keypoints = bbox_3d_keypoints or []
        # 每条车道线用二次曲线 [a,b,c] 表示 y=ax²+bx+c
        self.lane_coeffs = lane_coeffs or []
        # 相机内参+畸变
        self.cam_params = cam_params or {"K": np.eye(3).tolist(), "dist": [0.0]*5}
        # 场景元数据
        self.meta = meta or {"scene_type": "urban", "light": "day", "view_angle": 30}

    def to_dict(self):
        return {
            "bbox_3d_keypoints": self.bbox_3d_keypoints,
            "lane_coeffs": self.lane_coeffs,
            "cam_params": self.cam_params,
            "meta": self.meta
        }

    @classmethod
    def from_dict(cls, data: Dict):
        return cls(
            bbox_3d_keypoints=data.get("bbox_3d_keypoints"),
            lane_coeffs=data.get("lane_coeffs"),
            cam_params=data.get("cam_params"),
            meta=data.get("meta")
        )

# --- 基础配置 ---

TASK_DESCRIPTION_V2 = """
Create an autonomous driving scene image with structured annotations:
1. Lane Boundaries: Represented by quadratic curves (y=ax²+bx+c).
2. 3D Bounding Boxes: Defined by 8 precise corner keypoints.
3. Scene Context: {scene_type} under {light} lighting.
"""

# 审核阈值与目标
EVAL_CONFIG = {
    "target_score": 0.85,
    "max_iters": 10,
    "min_gain_threshold": 0.01,
    "pck_threshold": 5.0  # 像素误差容忍度
}
