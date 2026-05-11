import json
import os
import time
from datetime import datetime
from src.api.qwen_client import QwenClient, ImageGenClient
from src.utils.config import TASK_DESCRIPTION, EVALUATION_PROMPT_TEMPLATE

def run_experiment_2():
    print("开始实验2：固定 Seed=42, prompt_extend=False, 10次采样，迭代2次")
    
    # 1. 加载第 11 次迭代的基准 Prompt
    with open("outputs/results.json", "r", encoding="utf-8") as f:
        history = json.load(f)
    
    current_prompt = None
    for entry in history:
        if entry["iteration"] == 11:
            current_prompt = entry["prompt"]
            break
    
    if not current_prompt:
        print("未找到第 11 次迭代的 Prompt，使用默认任务描述生成初始指令。")
        current_prompt = "Based on test.png, draw solid yellow lane lines on all lane edges and add 3D bounding boxes with 8 corner points for every car."

    qwen = QwenClient(model="qwen3.6-plus")
    image_gen = ImageGenClient(model="qwen-image-2.0-pro")
    image_path = "test.png"
    
    experiment_results = []

    for iteration in range(1, 3):  # 迭代两次
        print(f"\n==== 实验2 - 迭代 {iteration} ====")
        print(f"当前指令: {current_prompt}")
        
        iteration_samples = []
        for i in range(1, 11):  # 生成 10 张
            print(f"正在生成样本 {i}/10 (Seed=42, prompt_extend=False)...")
            try:
                # 调用编辑接口
                img_out = image_gen.edit_image(
                    image_path, 
                    current_prompt, 
                    iteration=f"实验2_iter{iteration}_sample{i}",
                    seed=42,
                    prompt_extend=False
                )
                print(f"样本 {i} 已保存: {img_out}")
                iteration_samples.append(img_out)
            except Exception as e:
                print(f"样本 {i} 生成失败: {e}")
            
            if i < 10:
                print("等待 30 秒以避免限流...")
                time.sleep(30)
        
        # 评估（取该组最后一张进行评估，用于驱动下一次迭代）
        if iteration_samples:
            last_img = iteration_samples[-1]
            print(f"\n正在对该组最后一张图片进行评估: {last_img}")
            eval_prompt = EVALUATION_PROMPT_TEMPLATE.format(task_description=TASK_DESCRIPTION)
            feedback = qwen.evaluate_image(last_img, eval_prompt)
            print(f"评估反馈: {feedback}")
            
            experiment_results.append({
                "experiment": "实验2",
                "iteration": iteration,
                "prompt": current_prompt,
                "samples": iteration_samples,
                "feedback": feedback
            })
            
            # 生成下一次迭代的 Prompt
            refine_request = f"""
            Task: {TASK_DESCRIPTION}
            Current Instruction: {current_prompt}
            Visual Feedback from Sample: {feedback}
            
            Please refine the instruction to better achieve the task (yellow lines + points-first 3D boxes).
            Only provide the refined instruction text.
            """
            current_prompt = qwen.generate_text(
                prompt=refine_request,
                system_prompt="You are an expert prompt engineer."
            )
            
            # 实时保存实验结果
            with open("outputs/experiment2_results.json", "w", encoding="utf-8") as f:
                json.dump(experiment_results, f, indent=4, ensure_ascii=False)
        else:
            print("本轮无成功样本，迭代终止。")
            break

    print("\n实验2完成。结果已保存至 outputs/experiment2_results.json")

if __name__ == "__main__":
    run_experiment_2()
