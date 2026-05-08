import os
import json
from src.api.qwen_client import QwenClient, ImageGenClient
from src.utils.config import TASK_DESCRIPTION, EVALUATION_PROMPT_TEMPLATE

class AutoPromptAgent:
    def evaluate(self, image_path, prompt):
        eval_prompt = EVALUATION_PROMPT_TEMPLATE.format(
            task_description=self.task_description
        )
        return self.qwen.evaluate_image(image_path, eval_prompt)

    def __init__(self, task_description=TASK_DESCRIPTION, max_iterations=5):
        self.task_description = task_description
        self.max_iterations = max_iterations
        self.qwen = QwenClient()
        self.image_gen = ImageGenClient()
        self.history = []

    def __init__(self, task_description=TASK_DESCRIPTION, max_iterations=5):
        self.task_description = task_description
        self.max_iterations = max_iterations
        self.qwen = QwenClient()
        self.image_gen = ImageGenClient()
        self.history_file = "outputs/results.json"
        self.history = self._load_history()

    def _load_history(self):
        """加载已有的历史记录"""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data if isinstance(data, list) else []
            except Exception as e:
                print(f"Error loading history: {e}")
        return []

    def save_history(self):
        """实时保存历史记录到 JSON 文件"""
        if not os.path.exists("outputs"):
            os.makedirs("outputs")
        with open(self.history_file, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=4, ensure_ascii=False)

    def run_optimization_loop(self, initial_image=None):
        """核心优化循环"""
        print(f"Starting optimization for task: {self.task_description}")
        
        # 确定起始迭代轮次
        start_iter = len(self.history)
        print(f"Resuming from iteration {start_iter + 1}")

        # 获取当前最新的反馈和 Prompt（如果有的话）
        last_feedback = ""
        current_prompt = ""
        
        if self.history:
            last_entry = self.history[-1]
            last_feedback = last_entry.get("feedback", "")
            # 基于最后的反馈生成新的指令
            refine_request = f"""
            Task: {self.task_description}
            Current Instruction/Prompt: {last_entry.get('prompt', '')}
            Feedback from Evaluation: {last_feedback}
            
            Based on the previous results, please provide a NEW instruction for the image editing model to further improve the result. 
            Focus specifically on the accuracy of yellow lane lines and the "points-to-box" methodology.
            Only provide the instruction text.
            """
            current_prompt = self.qwen.generate_text(
                prompt=refine_request,
                system_prompt="You are an expert prompt engineer. Refine the instruction based on visual feedback."
            )
        else:
            # 初始生成
            if initial_image:
                current_prompt = self.qwen.generate_text(
                    prompt=f"Task: {self.task_description}\nGiven the initial image, what instructions should I give to an image editing model to achieve the task? Provide only the instruction text.",
                    system_prompt="You are an expert at image editing instructions."
                )
            else:
                current_prompt = self.qwen.generate_text(
                    prompt=f"Task: {self.task_description}\nGenerate a highly detailed prompt for an image generation model to achieve this task.",
                    system_prompt="You are an expert prompt engineer for image generation models."
                )
        
        for i in range(self.max_iterations):
            iteration_num = start_iter + i + 1
            print(f"\n--- Iteration {iteration_num} ---")
            print(f"Current Prompt/Instruction: {current_prompt}")
            
            # 2. 生成或编辑图像
            try:
                if initial_image:
                    print(f"Editing original image: {initial_image}")
                    image_path = self.image_gen.edit_image(initial_image, current_prompt, iteration=iteration_num)
                else:
                    print("Generating new image from scratch...")
                    image_path = self.image_gen.generate_image(current_prompt, iteration=iteration_num)
                print(f"Image processed: {image_path}")
            except Exception as e:
                print(f"Image processing failed: {e}")
                break
            
            # 3. 评估图像
            feedback = self.evaluate(image_path, current_prompt)
            print(f"Feedback: {feedback}")
            
            # 保存历史记录
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            self.history.append({
                "iteration": iteration_num,
                "timestamp": timestamp,
                "prompt": current_prompt,
                "image_path": image_path,
                "feedback": feedback
            })
            self.save_history() # 实时保存
            
            # 4. 根据反馈修正 Prompt/Instruction
            refine_request = f"""
            Task: {self.task_description}
            Previous Action: {"Edited original image" if initial_image else "Generated image"}
            Current Instruction/Prompt: {current_prompt}
            Feedback from Evaluation: {feedback}
            
            Based on the feedback, please provide a NEW instruction for the image editing model to further improve the result. 
            Focus specifically on the accuracy of yellow lane lines and the "points-to-box" methodology.
            Only provide the instruction text.
            """
            current_prompt = self.qwen.generate_text(
                prompt=refine_request,
                system_prompt="You are an expert prompt engineer. Refine the instruction based on visual feedback."
            )
            
        return self.history

    def run_batch_variations(self, num_variations=10):
        """遍历不同类型的 Prompt"""
        print(f"Generating {num_variations} variations for task: {self.task_description}")
        
        variations_request = f"""
        Task: {self.task_description}
        Please generate {num_variations} different styles or types of prompts to achieve this task. 
        Each prompt should focus on different aspects (e.g., photorealistic, schematic, night view, etc.).
        Provide the output in a JSON list format: ["prompt1", "prompt2", ...]
        """
        
        response = self.qwen.generate_text(
            prompt=variations_request,
            system_prompt="You are an expert prompt engineer. Provide only JSON output."
        )
        
        try:
            # 清理可能存在的 markdown 代码块
            json_str = response.strip().replace('```json', '').replace('```', '')
            prompts = json.loads(json_str)
        except Exception as e:
            print(f"Failed to parse JSON: {e}")
            return []

        results = []
        for i, prompt in enumerate(prompts):
            print(f"Processing variation {i+1}/{num_variations}: {prompt}")
            try:
                image_path = self.image_gen.generate_image(prompt)
                feedback = self.evaluate(image_path, prompt)
                results.append({
                    "prompt": prompt,
                    "image_path": image_path,
                    "feedback": feedback
                })
            except Exception as e:
                print(f"Error in variation {i+1}: {e}")
                
        return results
