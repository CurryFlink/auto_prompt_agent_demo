import os
import dashscope
from dashscope import ImageSynthesis
from http import HTTPStatus
from dotenv import load_dotenv
import requests

load_dotenv()
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

class ImageGenClient:
    def __init__(self, model="wanx-v1", edit_model="qwen-image-edit"):
        self.model = model
        self.edit_model = edit_model

    def generate_image(self, prompt, output_dir="outputs"):
        """调用文生图 API 生成图像并保存"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        rsp = ImageSynthesis.call(
            model=self.model,
            prompt=prompt,
            n=1,
            size='1024*1024'
        )
        
        if rsp.status_code == HTTPStatus.OK:
            return self._save_image(rsp.output.results[0].url, output_dir, "gen")
        else:
            raise Exception(f"Error generating image: {rsp.code} - {rsp.message}")

    def edit_image(self, image_path, prompt, output_dir="outputs"):
        """使用 Qwen-Image-Edit 编辑现有图像"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # qwen-image-edit 通常需要本地文件路径或 URL
        # SDK 中可能需要以 file:// 开头
        abs_path = os.path.abspath(image_path)
        
        rsp = ImageSynthesis.call(
            model=self.edit_model,
            prompt=prompt,
            image=f"file://{abs_path}",
            n=1,
            size='1024*1024'
        )

        if rsp.status_code == HTTPStatus.OK:
            return self._save_image(rsp.output.results[0].url, output_dir, "edit")
        else:
            raise Exception(f"Error editing image: {rsp.code} - {rsp.message}")

    def _save_image(self, image_url, output_dir, prefix):
        """内部辅助函数：下载并保存图像"""
        import time
        image_name = f"{prefix}_{int(time.time())}.png"
        image_path = os.path.join(output_dir, image_name)
        
        img_data = requests.get(image_url).content
        with open(image_path, 'wb') as handler:
            handler.write(img_data)
            
        return image_path
