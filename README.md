# Qwen2.5-VL 工程图纸元件关联识别

基于 Qwen2.5-VL-7B-Instruct 多模态模型，通过 LoRA 微调，自动识别工程 CAD 图纸中元件之间的从属关联关系。

> 详细项目文档见 [PROJECT.md](PROJECT.md)。

---

## 快速开始

### 环境准备

```bash
pip install -r requirements.txt
```

### 数据预处理

```bash
python preprocess.py --config config.yaml
```

输出到 `processed_data/` 目录，包含裁剪图片 + 关联关系 JSON。

### 训练

```bash
# 单卡
python train.py --config config.yaml

# 多卡 DDP
bash train_ddp.sh

# 从 checkpoint 恢复
python train.py --resume_from_checkpoint output/checkpoint-1500
```

### 推理服务部署

```bash
cd app
# 1. 启动 vLLM
python start_vllm_server.py &

# 2. 启动 Flask
python app.py
```

服务默认在 `http://localhost:5004`，接口参考 [`app/README.md`](app/README.md)。

### 端到端评估

```bash
cd app
python evaluate_associations.py
```

使用 DWG 审核标注数据作为 GT，自动调 API 推理后计算指标。

---

## 项目结构

| 模块 | 文件 | 说明 |
|---|---|---|
| **预处理** | `preprocess.py` | 数据预处理，裁剪+标注+生成样本 |
| **数据集** | `dataset.py` | PyTorch Dataset、Prompt 构建 |
| **训练** | `train.py`, `train_ddp.sh` | LoRA 微调 |
| **评估** | `evaluate.py` | 模型评估（微调模型直接推理） |
| **推理** | `app/app.py` | Flask + vLLM 推理接口 |
| **端到端评估** | `app/evaluate_associations.py` | 调 API 评估，对标标注 GT |
| **图服务** | `app/vlm_graph_service.py` | VLM 整图构建服务 |

## 关联关系类型

| 类型 | 说明 | API 支持 |
|---|---|---|
| `device-tag` | 设备 → 位号 | ✓ |
| `device-number` | 设备 → 设备序号 | ✓ |
| `symbol-tag` | 阀门符号 → 位号 | ✓ |
| `symbol-attr` | 阀门符号 → 属性标签 | ✓ |
| `instrument-tag` | 仪表 → 位号 | ✗ |

## 配置

编辑 `config.yaml`（训练）或 `app/config.yaml`（推理服务），支持路径、模型、训练参数、LoRA 参数等配置。

## 技术栈

- **模型：** Qwen2.5-VL-3B/7B-Instruct
- **微调：** PEFT (LoRA) + Transformers Trainer
- **推理：** vLLM + Flask
- **多卡：** DDP / device_map="auto"

## License

本项目基于 Qwen2.5-VL 开源模型，遵循其许可协议。
