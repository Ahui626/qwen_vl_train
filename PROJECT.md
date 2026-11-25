# 工程CAD图纸元件关联识别项目

## 项目概述

基于 **Qwen2.5-VL-7B-Instruct** 多模态大模型，通过 LoRA 微调，识别工程 CAD 图纸中元件之间的从属关联关系（如：阀门属于哪个位号、设备对应哪个编号、符号对应什么属性标签）。

输入为裁剪后的图纸局部图片（含红/黄色标注框）和元件的结构化描述，输出为关联目标的元件 ID。

---

## 关联关系类别

| 关联类型 | source | target | 说明 | API 支持 |
|---|---|---|---|---|
| `device-tag` | 设备 | 位号 | 设备对应的位号标注 | ✓ |
| `device-number` | 设备 | 设备序号 | 设备对应的编号标注 | ✓ |
| `symbol-tag` | 符号（阀门类） | 位号 | 符号对应的位号标注 | ✓ |
| `symbol-attr` | 符号（阀门类） | 属性标签 | 符号对应的属性（如公称直径） | ✓ |
| `instrument-tag` | 仪表 | 位号 | 仪表对应的位号标注 | ✗（API 过滤） |

**排除规则：**
- 设备—属性标签（不合理）
- 符号—设备序号（不合理）
- 仪表相关关联（app.py 的推理流程中过滤）
- 主体类元件之间互相关联（设备—设备、符号—符号等）
- 端口、V/P/Ø 开头的属性标签不作为候选

---

## 数据准备

### 1. 原始数据格式

每个项目文件夹包含：
- 一张工程 CAD 图纸（`.jpg`）
- 元件标注文件（`.txt`）：每行格式 `id,category,sub_category,x1,y1,x2,y2,...`（世界坐标）
- 关联关系文件（`*_关联.txt`）：每行两个元件 ID
- 审核后的从属关系（`dwg_edited.json`）：包含 nodes + edges，可以作为评估 GT

### 2. 数据预处理（`preprocess.py`）

```bash
python preprocess.py --config config.yaml
```

**处理步骤：**
1. 加载元件（过滤有效类别：设备、符号、位号、设备序号、属性标签、仪表）
2. 世界坐标 → 像素坐标
3. 过滤仪表相关关联
4. 构建关联图，以 **符号/设备为主体** 收集其关联的位号/设备序号/属性标签

**裁剪方式（两种）：**

| 类型 | 触发条件 | 裁剪方法 |
|---|---|---|
| `device_expanded` | 设备包围框内有内部位号/设备序号 | 以设备+关联元件的 bbox 范围+padding=50 扩展裁剪，最小 1024×1024 |
| `symbol_centered` | 阀门符号、设备内无内部元件 | 以主体为中心，固定 1024×1024 正方形裁剪 |

**标注规则：**
- 红色框（ID=1）：主体符号/设备
- 黄色框（ID=2,3,...）：候选元件，按与主体的边界距离排序，最多 7 个
- 模型输出关联的黄色框 ID，无关联输出 `NONE`

### 3. 关联标注数据（DWG 审核数据）

```
测试集/
└── 10个项目_项目1_系统图_精处理系统图 - 酸碱再生/
    ├── *.jpg                     # 图纸图片
    ├── *_edited.json             # 审核后的节点与边（GT）
    └── ...                       # 其他标注文件
```

`dwg_edited.json` 结构：
```json
{
  "nodes": [
    {"ele_id": "xxx", "label": "设备", "cate": "设备", "bbox": [x1,y1,x2,y2]},
    {"ele_id": "yyy", "label": "符号", "cate": "闸阀", "bbox": [x1,y1,x2,y2]}
  ],
  "edges": [
    {"con_id": "con001", "connection": "从属", "ele_id1": "xxx", "ele_id2": "yyy"}
  ]
}
```

其中 `connection="从属"` 的边即为关联关系 GT，根据两端的 `label` 确定关系类型。

---

## 模型训练

### 训练流程

```bash
# 单卡
python train.py --config config.yaml

# 多卡 DDP
bash train_ddp.sh

# 从 checkpoint 恢复
python train.py --resume_from_checkpoint output/checkpoint-1500
```

**训练配置：**
- 基础模型：`Qwen2.5-VL-7B-Instruct`
- 微调方式：LoRA（r=8, alpha=16, dropout=0.1）
- 目标模块：`q_proj`, `k_proj`, `v_proj`, `out_proj`, `down_proj`
- 等效 batch_size=16（gradient_accumulation_steps=16）
- 学习率 1e-4，warmup_ratio=0.05
- 训练/验证集比例 9:1

**训练 Prompt：**

以红色框主体（设备/符号）为中心，候选元件（位号/属性标签/设备序号）为黄色框，模型需要输出关联的黄色框 ID（逗号分隔）或 `NONE`。

### 训练输出

```
output_all_tag/
├── final/                        # 最终模型（LoRA权重）
├── checkpoint-1000/              # 中间 checkpoint
├── training_curves.png           # 训练曲线
├── training_log.jsonl            # 详细日志
└── evaluation_results.json       # 评估结果
```

---

## 推理服务

项目提供两套推理方案：

### 方案一：Flask + vLLM 服务（`app/app.py`）

```bash
# 1. 先启动 vLLM 服务
cd app
python start_vllm_server.py

# 2. 再启动 Flask 服务
python app.py
```

**架构：**
```
Flask (端口 5004) → vLLM (端口 5003) → GPU 推理
```

**主体推理流程：**
1. 接收图片 + 元件 classifications + 世界坐标
2. 坐标转换（世界坐标 → 像素坐标）
3. 筛选主体：设备 + subclass 含"阀"的符号
4. 对每个主体：选择候选 → 裁剪 → 画框 → 构建 Prompt → 调用 vLLM
5. 解析输出 → 过滤不合理关联 → 计算置信度

**API：**

| 路由 | 方法 | 说明 |
|---|---|---|
| `/health` | GET | 健康检查 |
| `/calculate_associations` | POST | 关联关系推理 |

**请求格式（multipart/form-data）：**

| 字段 | 类型 | 说明 |
|---|---|---|
| `image` | file | 图片文件 |
| `classifications` | JSON string | 元件列表 |
| `element_lists` | JSON string | 原始图元（接口忽略） |
| `x_min/y_min/x_max/y_max` | float | 世界坐标范围 |

```json
// classifications 格式
[{
  "id": "PE00001",
  "category": "设备",
  "subclass": "换热器",
  "x1": -1000.0, "y1": -500.0,
  "x2": -800.0,  "y2": -300.0
}]
```

**返回格式：**
```json
{
  "success": true,
  "associations": [
    {"source": "PE00001", "target": "PE00010", "type": "device-tag", "conf": 0.923}
  ],
  "stats": {...}
}
```

### 方案二：VLM 图构建服务（`app/vlm_graph_service.py`）

基于 VLM 对整个图片进行图构建，直接输出 nodes + edges，支持更多关联类型（含 instrument-tag）。

```bash
python vlm_graph_service.py
```

### 服务对比

| 对比项 | Flask + vLLM (app.py) | VLM 图服务 |
|---|---|---|
| 推理粒度 | 每个主体逐个推理 | 整图一次推理 |
| instrument-tag | 不支持（过滤） | 支持 |
| 输出格式 | associations 列表 | nodes + edges 图结构 |
| 服务端口 | 5004 | 5004（独立运行） |

---

## 评估

### 端到端评估（`evaluate_associations.py`）

使用 DWG 审核数据作为 GT，调用 API 推理后比较：

```bash
cd app/
python evaluate_associations.py              # 完整评估
python evaluate_associations.py --report     # 仅 GT 统计
python evaluate_associations.py --dry-run    # 模拟运行
```

**数据路径（硬编码）：**
```
/mnt/nvme/qwen_train/app/tmp_test/_原理图训练集（第一批次）/
├── 精处理系统图 - 运行/
├── 精处理系统图 - 酸碱再生/
├── 天明电厂03--精处理系统系统 - 再生/
└── ...
```

**输出：**
- 每个子文件夹的 P/R/F1 指标
- 按类型的汇总指标
- `evaluation_results.json`

### 模型评估（`evaluate.py`）

对预处理后的测试数据，直接加载微调模型推理：

```python
python evaluate.py
```

在代码顶部配置模型和数据路径，支持批处理推理。

### 标注数据 GT 分布

| 类型 | 数量 | API 支持 |
|---|---|---|
| symbol-attr | 953 | ✓ |
| symbol-tag | 895 | ✓ |
| line-label | 625 | - |
| line-direction | 428 | - |
| instrument-tag | 222 | ✗ |
| device-tag | 130 | ✓ |
| device-number | 74 | ✓ |
| 其他 | 128 | - |
| **合计** | **3455** | |

---

## 目录结构

```
qwen_train/
├── config.yaml                  # 训练配置文件
├── train.py                     # 训练主脚本
├── dataset.py                   # 训练数据集 & Prompt 构建
├── preprocess.py                # 数据预处理
├── evaluate.py                  # 模型评估
├── merge_knn7_data.py           # 合并 KNN 候选数据
├── plot_training_metrics.py     # 训练曲线绘制
├── test_model.py                # 模型快速测试
├── visualize.py                 # 可视化
├── requirements.txt             # Python 依赖
├── train_ddp.sh                 # 多卡 DDP 训练脚本
│
├── processed_data_train_tag/    # 预处理后的训练数据
├── processed_data_test_tag/     # 预处理后的测试数据
│
├── output_all_tag/              # 训练输出
│
├── app/                         # 推理服务
│   ├── app.py                   # Flask 推理接口
│   ├── utils.py                 # 图片处理、Prompt、结果解析
│   ├── config.yaml              # 推理服务配置
│   ├── vlm_graph_service.py     # VLM 图构建服务
│   ├── line_extractor.py        # 线条提取工具
│   ├── start_vllm_server.py     # vLLM 启动脚本
│   └── evaluate_associations.py # 端到端评估脚本
│
├── collect/                     # 实验收集数据
├── compare_knn_qwen/            # KNN vs Qwen 对比实验
├── test_data/                   # 测试数据
├── badcase_analysis/            # Badcase 分析
│
├── PROJECT.md                   # 本文档
└── README.md                    # 快速入门
```

---

## 技术栈

- **模型：** Qwen2.5-VL-7B-Instruct
- **微调：** PEFT (LoRA)
- **框架：** Transformers, Hugging Face Trainer
- **推理：** vLLM, Flask
- **多卡：** DDP, device_map="auto"
- **数据：** PIL, OpenCV, NumPy
