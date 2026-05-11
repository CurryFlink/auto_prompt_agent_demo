import json
import os
import time
from datetime import datetime
from src.api.qwen_client import QwenClient, ImageGenClient
from src.utils.config import TASK_DESCRIPTION, EVALUATION_PROMPT_TEMPLATE

def run_experiment_3():
    print("开始实验3：基于 nuScenes 真值投影的闭环优化 (3图/2采样/5轮)")
    
    # 加载 nuScenes 元数据（包含 2D 投影真值）
    try:
        with open("nusc_meta.json", "r", encoding="utf-8") as f:
            nusc_data = json.load(f)
    except Exception as e:
        print(f"错误: 无法加载 nusc_meta.json: {e}")
        return

    # 明确选择 nusc_test_1, 3, 5 三张图片进行训练
    target_files = ["nusc_test_1.png", "nusc_test_3.png", "nusc_test_5.png"]
    selected_data = [item for item in nusc_data if item['filename'] in target_files]
    base_images = [item['filename'] for item in selected_data]
    gt_map = {item['filename']: item for item in selected_data}

    # 加载初始 Prompt
    current_prompt = None
    if os.path.exists("outputs/results.json"):
        with open("outputs/results.json", "r", encoding="utf-8") as f:
            history = json.load(f)
            if history:
                current_prompt = history[-1]["prompt"]
    
    if not current_prompt:
        current_prompt = "Draw solid yellow lane lines on road edges and add 3D bounding boxes with 8 corner dots for every car. Ensure high geometric accuracy."

    qwen = QwenClient(model="qwen3.6-plus")
    image_gen = ImageGenClient(model="qwen-image-2.0-pro")
    
    experiment_results = []
    output_file = "outputs/experiment3_results.json"

    for iteration in range(1, 6):  # 5 轮迭代
        print(f"\n==== 实验3 - 迭代 {iteration}/5 ====")
        print(f"当前指令: {current_prompt}")
        
        iteration_data = {
            "iteration": iteration,
            "prompt": current_prompt,
            "scenarios": []
        }
        
        all_feedback_texts = []
        
        for img_idx, base_img in enumerate(base_images):
            meta = gt_map[base_img]
            print(f"\n正在处理场景 {img_idx+1}/3: {meta['scene_description']}")
            
            samples = []
            for s in range(1, 3):  # 每张图 2 个采样
                print(f"  正在生成采样 {s}/2 (Seed=42, prompt_extend=False)...")
                try:
                    img_out = image_gen.edit_image(
                        base_img, 
                        current_prompt, 
                        iteration=f"实验3_iter{iteration}_img{img_idx+1}_sample{s}",
                        seed=42,
                        prompt_extend=False
                    )
                    print(f"  采样 {s} 已保存: {img_out}")
                    samples.append(img_out)
                except Exception as e:
                    print(f"  采样 {s} 生成失败: {e}")
                
                time.sleep(30)
            
            # 精确评估环节：利用真值投影
            if samples:
                eval_img = samples[0]
                print(f"  正在进行量化评估: {eval_img}")
                
                # 1. 首先让 Qwen-VL 提取图中画出的点
                extract_prompt = "Identify all 3D bounding boxes you drew in this image. For each box, list the 8 corner coordinates in JSON format: {'boxes': [[(x1,y1),...], ...]}"
                vision_res = qwen.evaluate_image(eval_img, extract_prompt)
                
                # 2. 构建对比反馈 (Qwen 作为优化器，对比‘生成的点’与‘真值点’)
                # 注意：我们不把真值点直接给生成模型，只给优化器看
                refine_feedback_prompt = f"""
                Analysis Task: Compare the generated annotations in the image with the Ground Truth data.
                
                [Ground Truth (Projected 2D Corners for vehicles)]:
                {json.dumps(meta['gt_annotations'][:3], indent=2)} (Showing first 3 vehicles)
                
                [Model Extracted Points from Image]:
                {vision_res}
                
                Please provide a quantitative evaluation:
                1. How many cars were missed?
                2. What is the average pixel displacement of the 8 corners compared to GT?
                3. Are the yellow lines correctly placed on the lane boundaries?
                
                Summarize what the prompt needs to improve to reduce these specific geometric errors.
                """
                
                feedback = qwen.generate_text(refine_feedback_prompt, system_prompt="You are a geometric error analyzer.")
                print(f"  量化反馈摘要: {feedback[:150]}...")
                
                iteration_data["scenarios"].append({
                    "base_image": base_img,
                    "samples": samples,
                    "feedback": feedback,
                    "gt_data_used": True
                })
                all_feedback_texts.append(f"Scenario {base_img} Error Analysis: {feedback}")

        # 汇总反馈优化 Prompt
        combined_feedback = "\n".join(all_feedback_texts)
        refine_request = f"""
        Goal: Optimize the image generation prompt to achieve Zero Geometric Error for autonomous driving annotations.
        Current Instruction: {current_prompt}
        
        Consolidated Error Analysis from 3 nuScenes scenarios:
        {combined_feedback}
        
        Refine the instruction to be more prescriptive about geometry and coverage. Do NOT include coordinates in the final instruction.
        Only provide the refined instruction text.
        """
        
        print("\n正在根据量化误差进化 Prompt...")
        current_prompt = qwen.generate_text(
            prompt=refine_request,
            system_prompt="You are a master prompt engineer for precision geometry."
        )
        
        experiment_results.append(iteration_data)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(experiment_results, f, indent=4, ensure_ascii=False)

    print(f"\n实验3完成。结果已保存至 {output_file}")

    print(f"\n实验3完成。结果已保存至 {output_file}")

if __name__ == "__main__":
    run_experiment_3()
