import os
import requests
import base64
from openai import OpenAI
from http import HTTPStatus
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("DASHSCOPE_API_KEY")

class QwenClient:
    def __init__(self, model="qwen3.6-plus"):
        self.model = model
        # 使用 OpenAI 兼容模式
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

    def generate_text(self, prompt, system_prompt="You are a helpful assistant."):
        """使用 OpenAI 兼容模式调用 qwen3.6-plus"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in generate_text: {e}")
            raise

    def evaluate_image(self, image_path, prompt):
        """使用 OpenAI 兼容模式调用 qwen3.6-plus 进行图像理解"""
        try:
            print(f"DEBUG: Calling evaluate_image for {image_path}...")
            # 将图片转换为 base64
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                }
                            },
                        ],
                    }
                ],
            )
            print(f"DEBUG: Received response from evaluate_image.")
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in evaluate_image: {e}")
            raise

class ImageGenClient:
    def __init__(self, model="qwen-image-2.0-pro"):
        self.model = model
        self.api_key = api_key
        self.url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"

    def generate_image(self, prompt, output_dir="outputs", iteration=None, seed=None, prompt_extend=True):
        """使用标准 API 格式调用 qwen-image-2.0-pro 生成图像"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        data = {
            "model": self.model,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"text": prompt}]
                    }
                ]
            },
            "parameters": {
                "n": 1,
                "size": "1024*1024",
                "prompt_extend": prompt_extend
            }
        }
        if seed is not None:
            data["parameters"]["seed"] = seed

        response = requests.post(self.url, headers=headers, json=data)
        if response.status_code == 200:
            result = response.json()
            try:
                # 尝试多种可能的路径
                image_url = None
                if 'output' in result:
                    output = result['output']
                    if 'choices' in output:
                        choice = output['choices'][0]
                        if 'message' in choice and 'content' in choice['message']:
                            content = choice['message']['content']
                            if isinstance(content, list) and len(content) > 0:
                                image_url = content[0].get('image') or content[0].get('image_url')
                        if not image_url:
                            image_url = choice.get('image_url') or choice.get('url')
                    elif 'results' in output:
                        image_url = output['results'][0].get('url') or output['results'][0].get('image_url')
                
                if not image_url:
                    print(f"Response received but no URL found: {result}")
                    raise KeyError("image_url")
                    
                return self._save_image(image_url, output_dir, "gen", iteration)
            except Exception as e:
                print(f"Error parsing generate_image response: {e}. Full response: {result}")
                raise
        else:
            raise Exception(f"Error in generate_image: {response.text}")

    def edit_image(self, image_path, prompt, output_dir="outputs", iteration=None, seed=None, prompt_extend=True):
        """使用标准 API 格式调用 qwen-image-2.0-pro 编辑图像"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        data = {
            "model": self.model,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"image": f"data:image/png;base64,{base64_image}"},
                            {"text": prompt}
                        ]
                    }
                ]
            },
            "parameters": {
                "n": 1,
                "size": "1024*1024",
                "prompt_extend": prompt_extend
            }
        }
        if seed is not None:
            data["parameters"]["seed"] = seed

        response = requests.post(self.url, headers=headers, json=data)
        if response.status_code == 200:
            result = response.json()
            try:
                # 尝试多种可能的路径
                image_url = None
                if 'output' in result:
                    output = result['output']
                    if 'choices' in output:
                        choice = output['choices'][0]
                        if 'message' in choice and 'content' in choice['message']:
                            content = choice['message']['content']
                            if isinstance(content, list) and len(content) > 0:
                                image_url = content[0].get('image') or content[0].get('image_url')
                        if not image_url:
                            image_url = choice.get('image_url') or choice.get('url')
                    elif 'results' in output:
                        image_url = output['results'][0].get('url') or output['results'][0].get('image_url')

                if not image_url:
                    print(f"Response received but no URL found: {result}")
                    raise KeyError("image_url")

                return self._save_image(image_url, output_dir, "edit", iteration)
            except Exception as e:
                print(f"Error parsing edit_image response: {e}. Full response: {result}")
                raise
        else:
            raise Exception(f"Error in edit_image: {response.text}")

    def _save_image(self, image_url, output_dir, prefix, iteration=None):
        import time
        from datetime import datetime
        
        # 使用时间戳避免覆盖，如果提供 iteration 则包含它以便识别
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if iteration is not None:
            image_name = f"{prefix}_iter{iteration}_{timestamp}.png"
        else:
            image_name = f"{prefix}_{timestamp}.png"
            
        image_path = os.path.join(output_dir, image_name)
        
        img_data = requests.get(image_url).content
        with open(image_path, 'wb') as handler:
            handler.write(img_data)
            
        return image_path
