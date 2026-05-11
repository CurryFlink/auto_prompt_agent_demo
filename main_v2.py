import os
import json
from src.utils.schema import Annotation
from src.agent.prompt_agent_v2 import AutoPromptAgentV2

def run_v2_test():
    print("Initializing V2 Closed-Loop Test...")
    
    # 模拟模块 1：从‘先验库’中检索出的初始标注
    initial_anno = Annotation(
        bbox_3d_keypoints=[(450, 600), (550, 600), (550, 700), (450, 700), 
                           (450, 500), (550, 500), (550, 600), (450, 600)], # 一个粗略的立方体
        lane_coeffs=[[0.0001, 0.1, 800], [0.0001, -0.1, 200]], # 两条车道线
        meta={"scene_type": "highway", "light": "day", "view_angle": 30}
    )

    agent = AutoPromptAgentV2(initial_annotation=initial_anno)
    
    # 使用 test.png 作为基准图
    if os.path.exists("test.png"):
        agent.run_loop(initial_image="test.png")
    else:
        print("Error: test.png not found. Please ensure the base image exists.")

if __name__ == "__main__":
    run_v2_test()
