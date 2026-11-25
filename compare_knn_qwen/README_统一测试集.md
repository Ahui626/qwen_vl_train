# KNN和Qwen评估统一测试集

## 目的

为了公平对比KNN和Qwen模型的性能，需要确保两者使用相同的测试集和ground truth标准。

## 主要差异

### 原始测试集问题
- KNN测试集可能包含仪表相关的关联
- 没有包含无关联的阀门元件样本
- Ground truth与Qwen训练/测试数据不一致

### 统一后的测试集
- ✅ 去除所有仪表相关的关联关系
- ✅ 保留所有元件（包括无关联的阀门）
- ✅ 与Qwen的`processed_data_test/`使用相同的过滤规则

## 使用步骤

### 1. 生成统一测试集

首次使用前，运行脚本生成统一的测试集：

```bash
cd D:\qwen_train\compare_knn_qwen
python prepare_unified_testset.py
```

**输出**：
- 创建 `测试集_统一GT/` 目录
- 复制所有元件和图片
- 过滤仪表关联后保存新的`_关联.txt`文件
- 生成 `processing_summary.json` 统计报告

**统计示例**：
```
处理文件夹数: 4
原始关联总数: 940
过滤后关联数: 850
移除关联数: 90 (仪表相关)
总阀门元件数: 120
无关联阀门数: 15
```

### 2. 评估KNN模型

确保KNN服务运行后，评估KNN模型：

```bash
# 启动KNN服务（如果还没启动）
# python app.py  # 在旧的KNN接口目录运行

cd D:\qwen_train\compare_knn_qwen
python evaluate_knn_associations.py
```

**输出**：
- `results/knn_evaluation_summary.json`：KNN评估结果
- `intermediate_files/`：中间处理文件

### 3. 评估Qwen模型

Qwen模型评估使用`processed_data_test/`，该数据集已经应用了相同的过滤规则：

```bash
cd D:\qwen_train
python evaluate.py
```

**输出**：
- 评估指标输出到控制台
- `model_outputs_detailed.jsonl`：详细输出

### 4. 对比结果

现在两个模型的评估结果可以直接对比：

| 模型 | Precision | Recall | F1-Score | 说明 |
|------|-----------|--------|----------|------|
| KNN  | 从`knn_evaluation_summary.json`读取 | - | - | 基于距离的关联 |
| Qwen | 从`evaluate.py`输出读取 | - | - | 基于视觉语言模型 |

## 数据处理说明

### 过滤规则

参考 `preprocess.py` 中的逻辑：

1. **仪表过滤**：
   ```python
   # 过滤掉包含仪表的关联关系（line 477）
   relations = [r for r in relations 
                if r['element1_category'] != '仪表' 
                and r['element2_category'] != '仪表']
   ```

2. **阀门元件保留**：
   ```python
   # 为所有带"阀"字的符号创建样本，即使target_ids为空（line 501）
   valve_elements = [e for e in elements if is_valve_element(e)]
   for valve_elem in valve_elements:
       target_ids = main_to_targets.get(valve_id, [])  # 可能为空
       # 创建样本...
   ```

### 判断逻辑

```python
def is_valve_element(category: str, subclass: str) -> bool:
    """判断是否为阀门元件"""
    return category == '符号' and '阀' in subclass
```

## 目录结构

```
compare_knn_qwen/
├── 测试集/                        # 原始测试集（不修改）
│   ├── 项目1/
│   │   ├── 项目1.txt              # 元件信息
│   │   ├── 项目1_关联.txt          # 原始关联（含仪表）
│   │   ├── 项目1.jpg
│   │   └── 项目1.json
│   └── ...
├── 测试集_统一GT/                 # 统一后的测试集
│   ├── 项目1/
│   │   ├── 项目1.txt              # 元件信息（同原始）
│   │   ├── 项目1_关联.txt          # 过滤后的关联（无仪表）
│   │   ├── 项目1.jpg              # 复制的图片
│   │   ├── 项目1.json             # 复制的JSON
│   │   └── processing_summary.json # 处理统计
│   └── ...
├── prepare_unified_testset.py     # 生成统一测试集的脚本
├── evaluate_knn_associations.py   # KNN评估脚本（使用统一GT）
└── README_统一测试集.md           # 本文档
```

## 验证一致性

### 检查项

1. **关联数量一致性**：
   - 统计 `测试集_统一GT/` 中的关联总数
   - 对比 Qwen 测试集中非空`target_ids`的样本数

2. **仪表过滤**：
   - 确认 `测试集_统一GT/` 中的`_关联.txt`不包含仪表
   - 检查 `processing_summary.json` 中的`removed_associations`数量

3. **阀门元件**：
   - 确认所有带阀字的符号都在元件列表中
   - 检查是否有无关联的阀门（`unassociated_valves` > 0）

### 验证脚本示例

```python
import json
from pathlib import Path

# 读取统计信息
with open('测试集_统一GT/processing_summary.json', 'r', encoding='utf-8') as f:
    summary = json.load(f)

print(f"总文件夹: {summary['total_folders']}")
print(f"原始关联: {summary['total_original_associations']}")
print(f"过滤后: {summary['total_filtered_associations']}")
print(f"去除仪表: {summary['total_removed_associations']}")
print(f"无关联阀门: {summary['total_unassociated_valves']}")
```

## 常见问题

### Q: 为什么不直接修改原始测试集？
A: 保持原始数据完整性，方便后续对比和回溯。

### Q: 统一后的测试集是否需要重新生成？
A: 仅当原始测试集更新时才需要重新运行 `prepare_unified_testset.py`。

### Q: KNN和Qwen的评估指标还是不匹配？
A: 检查以下几点：
1. 确认两者使用相同的ground truth（关联数量应该一致）
2. 检查元素过滤规则（V/P/Ø属性标签、端口等）
3. 验证置信度阈值设置

### Q: 无关联的阀门元件如何评估？
A: 
- KNN：如果没有返回关联，则正确（TP=0, FP=0, FN=0 → P=1, R=1, F1=1）
- Qwen：如果输出"NONE"，则正确（同上）

## 参考代码

### preprocess.py 相关逻辑
- 第 477 行：过滤仪表关联
- 第 495-501 行：处理阀门元件（包括无关联）
- 第 379-381 行：`is_valve_element` 判断函数

### evaluate.py 相关逻辑
- 第 75-85 行：`parse_model_output` 处理"无"/"NONE"
- 第 118-130 行：`compute_metrics` 处理空关联的P/R/F1计算

## 更新记录

- 2026-02-10: 创建统一测试集脚本和文档
- 确保KNN和Qwen使用相同的ground truth标准
- 去除仪表关联，保留无关联阀门样本


