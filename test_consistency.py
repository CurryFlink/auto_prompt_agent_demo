import json
import os
from src.api.qwen_client import ImageGenClient

def run_test():
    # 1. 加载第 11 次迭代的 Prompt
    with open("outputs/results.json", "r", encoding="utf-8") as f:
        history = json.load(f)
    
    # 找到第 11 次的 prompt
    target_prompt = None
    for entry in history:
        if entry["iteration"] == 11:
            target_prompt = entry["prompt"]
            break
    
    if not target_prompt:
        print("未找到第 11 次迭代的 Prompt")
        return

    client = ImageGenClient()
    image_path = "test.png"
    
    # 第一组：原样生成 5 张
    print("开始生成第一组（原 Prompt）...")
    for i in range(5):
        print(f"正在生成第 {i+1} 张...")
        client.edit_image(image_path, target_prompt, iteration=f"consistency_group1_{i+1}")

    # 第二组：添加约束生成 5 张
    # 降低温度（虽然 REST API 不一定直接支持 temperature，但在 prompt 里强化一致性指令）
    consistency_prompt = target_prompt + "\n\nCRITICAL: Maintain absolute geometric consistency. Ensure the output is deterministic and identical in style and placement to the intended annotations. Minimize random variations."
    
    print("\n开始生成第二组（强化一致性 Prompt）...")
    import time
    for i in range(5):
        # 如果是重新运行，跳过已生成的（假设第1张已成功）
        if i == 0:
             print(f"正在生成第 {i+1} 张...")
             try:
                 client.edit_image(image_path, consistency_prompt, iteration=f"consistency_group2_{i+1}")
             except Exception as e:
                 print(f"第 {i+1} 张生成失败: {e}")
        elif i >= 1:
            print(f"正在等待 10 秒以避免限流，然后生成第 {i+1} 张...")
            time.sleep(10)
            try:
                client.edit_image(image_path, consistency_prompt, iteration=f"consistency_group2_{i+1}")
            except Exception as e:
                print(f"第 {i+1} 张生成失败: {e}")

    print("\n任务完成。请查看 outputs/ 目录下的 consistency_groupX 系列图片。")

if __name__ == "__main__":
    run_test()
