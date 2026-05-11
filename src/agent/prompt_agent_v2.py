import os
import json
import numpy as np
from datetime import datetime
from src.api.qwen_client import QwenClient, ImageGenClient
from src.utils.schema import Annotation, EVAL_CONFIG
from src.agent.evaluator import evaluate_geometry
from src.agent.prompt_engine import PromptEngine
from src.agent.control_logic import ControlLogic

class AutoPromptAgentV2:
    def __init__(self, initial_annotation: Annotation):
        self.qwen = QwenClient(model="qwen3.6-plus")
        self.image_gen = ImageGenClient(model="qwen-image-2.0-pro")
        self.prompt_engine = PromptEngine()
        self.control_logic = ControlLogic()
        
        self.current_anno = initial_annotation
        self.history = []
        self.history_file = "outputs/v2_results.json"

    def run_loop(self, initial_image: str):
        print(f"Starting V2 Closed-Loop Optimization...")
        
        # 初始 Meta 和参考
        meta = self.current_anno.meta
        ref_anno = self.current_anno.to_dict()
        deltas = {} # 初始增量为空

        for i in range(EVAL_CONFIG["max_iters"]):
            iteration_num = i + 1
            print(f"\n--- V2 Iteration {iteration_num} ---")
            
            # 1. 构建结构化 Prompt (模块 2 & 6)
            prompt = self.prompt_engine.build_prompt(meta, ref_anno, deltas)
            
            # 2. 生成/编辑图像 (模块 3)
            # 这里我们使用当前标注作为 prompt 的核心部分发送给生图模型
            try:
                # 模拟 ControlNet：将结构化标注转化为自然语言描述（或直接传 JSON 供模型理解）
                gen_prompt = f"Autonomous driving scene: {meta['scene_type']}. " \
                             f"Draw lane lines with coeffs {ref_anno['lane_coeffs']}. " \
                             f"Draw 3D boxes at {ref_anno['bbox_3d_keypoints']}."
                
                image_path = self.image_gen.edit_image(
                    initial_image, 
                    gen_prompt, 
                    iteration=f"v2_iter{iteration_num}",
                    seed=42,
                    prompt_extend=False
                )
                print(f"Image processed: {image_path}")
            except Exception as e:
                print(f"Image processing failed: {e}")
                break

            # 3. 几何指标审核 (模块 4)
            # 注意：实际场景中，这里需要从生成的图像中“反推”出 Annotation。
            # 暂由 Qwen3.6-plus 充当解析器，将视觉结果转回 Annotation 结构。
            parsed_anno_dict = self._vision_to_annotation(image_path)
            pred_anno = Annotation.from_dict(parsed_anno_dict)
            
            metrics = evaluate_geometry(pred_anno, self.current_anno)
            print(f"Metrics: {metrics}")

            # 4. 保存记录 (模块 7)
            entry = {
                "iteration": iteration_num,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "metrics": metrics,
                "annotation": pred_anno.to_dict(),
                "image_path": image_path
            }
            self.history.append(entry)
            self._save_history()

            # 5. 收敛与熔断判断 (模块 7)
            stop, reason = self._check_stop(iteration_num, metrics)
            if stop:
                print(f"Optimization stopped: {reason}")
                break

            # 6. 指标 -> Delta 映射 (模块 5)
            actions = self.control_logic.metrics_to_actions(metrics)
            
            # 7. LLM 采样下一轮 Delta (模块 6)
            refine_prompt = f"Current metrics: {metrics}. Action hints: {actions['hints']}. " \
                            f"Previous annotation: {ref_anno}. " \
                            f"Output new JSON deltas to improve geometry."
            
            llm_response = self.qwen.generate_text(refine_prompt, system_prompt="You are a geometry tuner.")
            deltas = self.prompt_engine.parse_llm_json(llm_response)
            
            # 更新当前参考标注为上一轮的最优（或累加 Delta）
            ref_anno = pred_anno.to_dict()

    def _vision_to_annotation(self, image_path: str) -> Dict:
        """
        利用 Qwen-VL 将像素结果转回结构化数据。
        这是闭环中‘感知’的关键一步。
        """
        prompt = "Extract 3D box 8 keypoints and lane quadratic coefficients from this image. Output JSON only."
        response = self.qwen.evaluate_image(image_path, prompt)
        return self.prompt_engine.parse_llm_json(response)

    def _check_stop(self, iteration: int, metrics: Dict) -> tuple:
        if metrics["total"] >= EVAL_CONFIG["target_score"]:
            return True, "target_reached"
        if iteration >= EVAL_CONFIG["max_iters"]:
            return True, "max_iters_reached"
        return False, ""

    def _save_history(self):
        with open(self.history_file, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=4, ensure_ascii=False)
