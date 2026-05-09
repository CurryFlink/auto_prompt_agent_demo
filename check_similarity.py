import os
from PIL import Image
import numpy as np

def calculate_similarity(img1_path, img2_path):
    """计算两张图片的像素级相似度"""
    try:
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        
        # 确保尺寸一致
        if img1.size != img2.size:
            img2 = img2.resize(img1.size)
            
        arr1 = np.array(img1).astype(float)
        arr2 = np.array(img2).astype(float)
        
        # 计算均方误差 (MSE)
        mse = np.mean((arr1 - arr2) ** 2)
        
        if mse == 0:
            return 100.0
            
        # 将 MSE 转换为百分比相似度（仅作为参考）
        # 最大可能的 MSE 是 255^2
        max_mse = 255.0 ** 2
        similarity = 100.0 * (1.0 - (mse / max_mse))
        return similarity
    except Exception as e:
        print(f"Error comparing {img1_path} and {img2_path}: {e}")
        return 0.0

def check_experiment_2_similarity():
    output_dir = "outputs"
    results = {}
    
    for iter_num in [1, 2]:
        print(f"\n--- 检查实验2 迭代 {iter_num} 的前 9 张图相似度 ---")
        
        # 找到基准图 (Sample 1)
        base_img_prefix = f"edit_iter实验2_iter{iter_num}_sample1_"
        base_img = None
        for f in os.listdir(output_dir):
            if f.startswith(base_img_prefix) and f.endswith(".png"):
                base_img = os.path.join(output_dir, f)
                break
        
        if not base_img:
            print(f"未找到迭代 {iter_num} 的 Sample 1")
            continue
            
        print(f"基准图: {os.path.basename(base_img)}")
        
        iter_similarities = []
        for s in range(2, 10):
            sample_prefix = f"edit_iter实验2_iter{iter_num}_sample{s}_"
            target_img = None
            for f in os.listdir(output_dir):
                if f.startswith(sample_prefix) and f.endswith(".png"):
                    target_img = os.path.join(output_dir, f)
                    break
            
            if target_img:
                sim = calculate_similarity(base_img, target_img)
                print(f"Sample {s} 相似度: {sim:.4f}%")
                iter_similarities.append(sim)
            else:
                print(f"未找到 Sample {s}")
        
        if iter_similarities:
            avg_sim = sum(iter_similarities) / len(iter_similarities)
            print(f"迭代 {iter_num} 平均相似度 (相对于 Sample 1): {avg_sim:.4f}%")

if __name__ == "__main__":
    check_experiment_2_similarity()
