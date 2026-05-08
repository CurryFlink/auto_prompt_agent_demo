# Auto Prompt Agent for Autonomous Driving Scenes

这是一个自动生成和优化图像生成提示词（Prompt）的 Agent，专门针对车道线和车辆 3D Box 绘制任务。

## 功能
- **循环优化模式 (Loop Mode)**: 自动生成 Prompt -> 调用文生图 API -> 调用 Qwen 评估图像 -> 根据反馈修正 Prompt -> 循环迭代。
- **批量变体模式 (Batch Mode)**: 让 Qwen 自动生成多种不同风格的 Prompt 并对比结果。
- **任务针对性**: 预置了针对自动驾驶场景（车道线、3D Box）的评估标准。

## 提示
注意config.py，这里有两个模板，会影响agent的行为。

## 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 配置 API Key
在项目根目录创建 `.env` 文件（或修改 `.env.template`）：
```
DASHSCOPE_API_KEY=你的阿里云DashScope_API_KEY
```

### 3. 运行
**循环优化模式（默认 3 次迭代）：**
```bash
python main.py --mode loop --iterations 3
```

**批量变体模式（生成 5 种不同尝试）：**
```bash
python main.py --mode batch --variations 5
```

## 项目结构
- `src/api/`: API 客户端（Qwen 文本/视觉、Wanx 文生图）
- `src/agent/`: 核心优化逻辑
- `src/utils/`: 配置和评估模板
- `outputs/`: 存储生成的图像和优化过程记录
