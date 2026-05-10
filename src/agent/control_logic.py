from typing import Dict, List, Any, Callable
import numpy as np

# --- 模块 5：指标 -> Delta 映射表 ---

class ControlLogic:
    def __init__(self):
        # 预定义规则
        self.rules = {
            "pck": [
                {"condition": lambda x: x < 0.5, "action": {"ref_weight": 0.9, "temp": 0.2, "hint": "Significant shift needed for keypoints."}},
                {"condition": lambda x: 0.5 <= x < 0.8, "action": {"ref_weight": 0.7, "temp": 0.4, "hint": "Fine-tune keypoints within ±5px."}},
            ],
            "box_integrity": [
                {"condition": lambda x: x < 0.7, "action": {"lock_aspect_ratio": True, "hint": "Correct the 3D volume proportions."}},
            ],
            "lane_consistency": [
                {"condition": lambda x: x < 0.6, "action": {"hint": "Align lane curvature (a-coefficient) to be more parallel."}},
            ]
        }

    def metrics_to_actions(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """将数值指标转化为 LLM 执行动作建议"""
        actions = {"hints": []}
        for metric, rule_list in self.rules.items():
            val = metrics.get(metric, 1.0)
            for rule in rule_list:
                if rule["condition"](val):
                    actions.update({k: v for k, v in rule["action"].items() if k != "hint"})
                    if "hint" in rule["action"]:
                        actions["hints"].append(rule["action"]["hint"])
                    break
        return actions

    def apply_momentum(self, current_delta: Dict, history: List[Dict], factor: float = 0.5) -> Dict:
        """加入动量记忆，平滑调整方向"""
        if not history:
            return current_delta
        
        # 简化处理：这里仅作为逻辑占位，实际应用中需要对像素坐标或系数进行加权
        # 在 V2 中，由 LLM 参考 history 指标自行实现动量效果更佳
        return current_delta
