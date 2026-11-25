# Qwen2.5-VL 推理接口

基于微调后的Qwen2.5-VL模型，提供元件关联关系推理服务。

## 快速启动

### 1. 配置

编辑 `config.yaml`，设置模型路径：

```yaml
model:
  # LoRA权重路径（训练输出的final目录）
  model_path: "/home/ubuntu/qwen_train/output_7b_merge/final"
  
  # 基础模型路径（必填）
  base_model_path: "/home/ubuntu/model/Qwen2.5-VL-7B-Instruct"
  
  # GPU设置（null则自动分配）
  device: null
```

**重要说明：**
- `model_path`: LoRA微调权重路径（训练时output_dir/final）
- `base_model_path`: Qwen2.5-VL基础模型路径（与训练时model.name一致）
- 两者缺一不可

### 2. 启动服务

**Windows:**
```bash
start.bat
```

**Linux/Mac:**
```bash
bash start.sh
```

服务将在 `http://localhost:5003` 启动。

## 接口说明

### 1. 健康检查

**请求：**
```
GET /health
```

**返回：**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "processor_loaded": true
}
```

### 2. 推理接口

**请求：**
```
POST /infer
或
POST /calculate_associations
```

**参数：**
- `image`: 图片文件（multipart/form-data）
- `classifications`: JSON字符串，元件列表
- `x_min`, `y_min`, `x_max`, `y_max`: 世界坐标系范围

**classifications格式：**
```json
[
  {
    "id": "PE0001",
    "category": "设备",
    "subclass": "泵",
    "x1": 1000.0,
    "y1": 1000.0,
    "x2": 1200.0,
    "y2": 1200.0,
    "content": ""
  },
  {
    "id": "PE0002",
    "category": "位号",
    "subclass": "位号",
    "x1": 1050.0,
    "y1": 1050.0,
    "x2": 1150.0,
    "y2": 1100.0,
    "content": "P-101"
  }
]
```

**返回：**
```json
{
  "success": true,
  "associations": [
    {
      "source": "PE0001",
      "target": "PE0002",
      "type": "device-tag",
      "conf": 0.9
    }
  ]
}
```

## 关联类型

| 类型 | 说明 | 主体元件 | 关联元件 |
|------|------|---------|---------|
| `device-tag` | 设备-位号 | 设备 | 位号 |
| `device-number` | 设备-设备序号 | 设备 | 设备序号 |
| `symbol-tag` | 符号-位号 | 符号 | 位号 |
| `symbol-attr` | 符号-属性标签 | 符号 | 属性标签 |

**注意：**
- 仪表类元件不参与关联关系计算
- 属性标签中content以V、P、Ø开头的不作为候选元件

## 配置说明

### 模型配置
- `model_path`: LoRA微调权重路径（训练输出的output/final）
- `base_model_path`: 基础模型路径（必填，如"/path/to/Qwen2.5-VL-7B-Instruct"）
- `device`: GPU设备（null则自动分配，或指定如"cuda:0"）
- `max_new_tokens`: 最大生成token数
- `temperature`: 采样温度
- `top_p`: 核采样参数

### 图片处理
- `max_size`: 最大图片尺寸（1024）
- `crop_size`: 裁剪尺寸（1024）
- `max_candidates`: 最大候选元件数（10）

### 关联关系
- `default_confidence`: 默认置信度（0.9）
- `category_map`: 类别映射（中文→英文）

## 注意事项

1. **内存要求**：7B模型需要约16GB显存
2. **启动时间**：首次启动需要加载模型，约30秒
3. **推理速度**：单个主体元件推理约1-2秒
4. **兼容性**：接口参数和返回值与旧版`calculate_associations`完全一致

### 过滤规则

以下类型的元件不参与关联关系计算：
- **仪表**：所有仪表类元件
- **特定属性标签**：content以V、P、Ø开头的属性标签

这些元件在裁剪图片时会被自动跳过。

## 目录结构

```
app/
├── config.yaml          # 配置文件
├── app.py               # Flask应用主文件
├── utils.py             # 工具函数
├── start.sh             # Linux启动脚本
├── start.bat            # Windows启动脚本
└── README.md            # 说明文档
```

