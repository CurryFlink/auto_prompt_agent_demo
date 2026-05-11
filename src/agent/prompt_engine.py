import json
import re
from typing import Dict, List, Any
from jinja2 import Template

# --- 模块 2：结构化 Prompt 模板 ---

PROMPT_TEMPLATE = """
Generate autonomous driving annotation for {{ meta.scene_type }} under {{ meta.light }} lighting.

Reference Geometry (Base State):
- BBOX Keypoints (pixel): {{ ref_annotation.bbox_3d_keypoints }}
- Lane Curves (y=ax²+bx+c): {{ ref_annotation.lane_coeffs }}

Apply strict delta adjustments based on feedback:
{% for key, delta in deltas.items() %}
- {{ key }}: {{ delta }}
{% endfor %}

[STRICT OUTPUT FORMAT]
You must output ONLY a valid JSON object containing the adjusted parameters. 
The keys MUST be: "bbox_3d_keypoints" and "lane_coeffs".
Example: {"bbox_3d_keypoints": [[10,20],...], "lane_coeffs": [[0.001, 0.1, 500],...]}
"""

class PromptEngine:
    def __init__(self):
        self.template = Template(PROMPT_TEMPLATE)

    def build_prompt(self, 
                     meta: Dict, 
                     ref_annotation: Dict, 
                     deltas: Dict) -> str:
        """根据元数据、参考标注和 Delta 增量构建 Prompt"""
        return self.template.render(
            meta=meta,
            ref_annotation=ref_annotation,
            deltas=deltas
        )

    def parse_llm_json(self, text: str) -> Dict[str, Any]:
        """使用正则提取并解析 LLM 输出的 JSON，防止幻觉"""
        try:
            # 提取第一个 { 和 最后一个 } 之间的内容
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                json_str = match.group()
                return json.loads(json_str)
            return {}
        except Exception as e:
            print(f"JSON Parsing Error: {e}, Text: {text}")
            return {}
