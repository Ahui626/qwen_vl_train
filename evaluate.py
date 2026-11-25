"""
模型评估工具
在测试集上计算准确率、召回率、F1分数等指标
"""
import os
import sys
import json
import random
import yaml
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ==================== 配置参数 ====================
# 直接在代码中指定路径，None 表示从配置文件读取

# 微调模型路径（LoRA模型）
# MODEL_PATH = "/home/ubuntu/su/qwen_train/output/final"  # 例如: "output/final" 或 "/home/ubuntu/qwen_train/output/checkpoint-1000"
MODEL_PATH = "/home/ubuntu/qwen_train/output_all_tag/final"  # 例如: "output/final" 或 "/home/ubuntu/qwen_train/output/checkpoint-1000"

# 预处理数据目录
# PROCESSED_DATA_DIR = "/home/ubuntu/qwen_train/test_output"  # 例如: "processed_data" 或 "/home/ubuntu/qwen_train/processed_data"
PROCESSED_DATA_DIR = "/home/ubuntu/qwen_train/processed_data_test_tag"  # 例如: "processed_data" 或 "/home/ubuntu/qwen_train/processed_data"

# 基础模型路径（可选，通常不需要改）
# BASE_MODEL_PATH = None  # 例如: "/home/ubuntu/model/Qwen2.5-VL-3B-Instruct"
BASE_MODEL_PATH = "/home/ubuntu/model/Qwen2.5-VL-7B-Instruct"  # 例如: "/home/ubuntu/model/Qwen2.5-VL-3B-Instruct"

# 评估输出目录（所有评估结果文件保存位置）
OUTPUT_DIR = "/home/ubuntu/qwen_train/evaluate_all_tag"  # 例如: "output" 或 "/home/ubuntu/qwen_train/evaluation_results"

# 批处理大小（用于加速评估）
EVAL_BATCH_SIZE = 8  # 可根据显存调整：2, 4, 8, 16, 32
# 显存参考：4→6GB, 8→10GB, 16→18GB, 32→35GB

# ===================================================

# 设置 GPU（在导入 torch 之前）
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES', '3')

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import PeftModel

from dataset import build_prompt, resize_image

# PyTorch优化设置
torch.backends.cudnn.benchmark = True  # 自动寻找最优算法
torch.set_float32_matmul_precision('high')  # 使用TF32提速（Ampere架构及以上）


def load_config(config_path: str = 'config.yaml') -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def parse_model_output(output: str) -> list:
    """
    解析模型输出的ID列表
    
    支持格式：
    - "3"
    - "2, 3"
    - "2,3"
    - "无" / "NONE" / "None" / "" (表示没有关联)
    """
    output = output.strip()
    
    # 处理空输出或"无"/"NONE"的情况
    if not output or output == "无" or output.upper() == "NONE":
        return []
    
    # 尝试解析逗号分隔的数字
    try:
        ids = []
        for part in output.replace(' ', '').split(','):
            part = part.strip()
            if part.isdigit():
                ids.append(int(part))
        return ids
    except:
        pass
    
    # 尝试解析单个数字
    try:
        if output.isdigit():
            return [int(output)]
    except:
        pass
    
    return []


def compute_metrics(pred_ids: list, gt_ids: list) -> dict:
    """
    计算评估指标（基于ID匹配）
    
    Args:
        pred_ids: 预测的ID列表 [2, 3]
        gt_ids: 真实的ID列表 [2, 3]
    
    特殊情况处理：
        当预测和真实都为空时（都正确预测"无关联"），precision=1.0, recall=1.0, f1=1.0
    """
    pred_set = set(pred_ids)
    gt_set = set(gt_ids)
    
    true_positive = len(pred_set & gt_set)
    false_positive = len(pred_set - gt_set)
    false_negative = len(gt_set - pred_set)
    
    # 特殊情况：预测和真实都为空（正确预测"无关联"）
    if len(pred_set) == 0 and len(gt_set) == 0:
        precision = 1.0
        recall = 1.0
        f1 = 1.0
    else:
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # 完全匹配（所有ID都对）
    exact_match = (pred_set == gt_set)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'exact_match': 1 if exact_match else 0,
        'true_positive': true_positive,
        'false_positive': false_positive,
        'false_negative': false_negative,
        'pred_count': len(pred_ids),
        'gt_count': len(gt_ids),
    }


def evaluate_model(model_path: str, data_dir: str, output_dir: str, max_samples: int = None, config: dict = None, use_validation_set: bool = False):
    """在预处理后的测试集上评估模型
    
    Args:
        model_path: 模型路径
        data_dir: 数据目录
        output_dir: 输出目录
        max_samples: 最大样本数（可选）
        config: 配置字典
        use_validation_set: 是否只使用验证集（默认False使用全量数据）
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(data_dir)
    
    # 加载配置（如果没有传入）
    if config is None:
        config = load_config()
    
    # 获取文件名配置
    paths_cfg = config.get('paths', {})
    samples_index_name = paths_cfg.get('samples_index', 'samples_index.json')
    evaluation_results_name = paths_cfg.get('evaluation_results', 'evaluation_results.json')
    
    print("="*60)
    print("模型评估")
    print("="*60)
    
    # 读取样本索引
    index_path = data_dir / samples_index_name
    if not index_path.exists():
        print(f"错误: 样本索引文件不存在: {index_path}")
        return None
    
    with open(index_path, 'r', encoding='utf-8') as f:
        all_samples = json.load(f)
    
    # 根据参数决定使用哪些数据
    if use_validation_set:
        # 使用验证集（与训练时的划分保持一致）
        train_ratio = config['training'].get('train_ratio', 0.9)
        random.seed(42)
        indices = list(range(len(all_samples)))
        random.shuffle(indices)
        split_idx = int(len(indices) * train_ratio)
        # 只使用验证集部分
        validation_indices = indices[split_idx:]
        test_samples = [all_samples[i] for i in validation_indices]
        print(f"使用验证集评估（训练数据的{int((1-train_ratio)*100)}%）")
    else:
        # 使用全量数据
        test_samples = all_samples
        print(f"使用全量数据评估")
    
    # 修正样本路径：samples_index.json 中的路径可能是相对路径
    # 需要使用实际的 data_dir 重新构建路径
    for sample in test_samples:
        json_path = Path(sample['json_path'])
        # 提取文件夹名和文件名
        # 例如：processed_data/10个项目_xxx/sample_0000.json
        # 提取：10个项目_xxx/sample_0000.json
        if len(json_path.parts) >= 2:
            # 获取最后两个部分：文件夹名/文件名
            folder_name = json_path.parts[-2]
            file_name = json_path.parts[-1]
            # 使用当前指定的 data_dir 重新构建完整路径
            sample['json_path'] = str(data_dir / folder_name / file_name)
        
        if 'image_path' in sample:
            image_path = Path(sample['image_path'])
            if len(image_path.parts) >= 2:
                folder_name = image_path.parts[-2]
                file_name = image_path.parts[-1]
                sample['image_path'] = str(data_dir / folder_name / file_name)
    
    print(f"评估样本数: {len(test_samples)}")
    
    # 优先使用代码中设置的基础模型路径，否则从配置文件读取
    base_model_path = BASE_MODEL_PATH if BASE_MODEL_PATH is not None else config.get('model', {}).get('name', '/home/ubuntu/model/Qwen2.5-VL-3B-Instruct')
    print(f"\n基础模型路径: {base_model_path}")
    
    # 从基础模型加载processor（不是从checkpoint）
    print(f"加载 processor from: {base_model_path}")
    processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
    # 设置padding方向为左侧（批处理时必须，避免警告）
    processor.tokenizer.padding_side = 'left'
    
    # 加载微调模型
    print(f"加载微调模型: {model_path}")
    
    base_model = AutoModelForVision2Seq.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,  # 使用FP16而不是BF16，更快
        device_map="auto",
        trust_remote_code=True,
        # load_in_8bit=True,  # 去掉8-bit量化，提升推理速度
    )
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()
    
    # 预加载所有数据（优化IO性能）
    print("预加载数据...")
    batch_data = []
    for sample in tqdm(test_samples, desc="加载数据"):
        with open(sample['json_path'], 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        image_path = Path(sample['json_path']).parent / data['image']
        image = Image.open(image_path).convert('RGB')
        image = resize_image(image)
        
        batch_data.append({
            'sample': sample,
            'data': data,
            'image': image,
        })
    
    # 打印前3个样本的详细信息用于调试
    print("\n" + "="*80)
    print("数据加载验证（前3个样本）")
    print("="*80)
    for i, item in enumerate(batch_data[:3], 1):
        data = item['data']
        print(f"\n样本 {i}: {Path(item['sample']['json_path']).name}")
        print(f"  主体元件: {data['main_element']}")
        print(f"  候选元件数: {len(data['candidate_elements'])}")
        print(f"  答案: {data['answer']}")
        print(f"  ground truth IDs: {data['answer'].get('target_ids', [])}")
        
        # 构建prompt看看
        question = build_prompt(data['main_element'], data['candidate_elements'])
        print(f"  Prompt长度: {len(question)} 字符")
        print(f"  Prompt前100字符: {question[:100]}...")
    print("="*80 + "\n")
    
    print(f"\n开始批处理推理 (batch_size={EVAL_BATCH_SIZE})...")
    
    # 批处理推理
    all_metrics = []
    sample_results = []
    processed_samples = 0
    
    for i in tqdm(range(0, len(batch_data), EVAL_BATCH_SIZE), desc="评估中"):
        batch = batch_data[i:i + EVAL_BATCH_SIZE]
        
        # 准备批次输入
        batch_messages = []
        batch_images = []
        batch_samples = []
        batch_datas = []
        
        for item in batch:
            data = item['data']
            question = build_prompt(data['main_element'], data['candidate_elements'])
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": question},
                    ],
                },
            ]
            batch_messages.append(messages)
            batch_images.append(item['image'])
            batch_samples.append(item['sample'])
            batch_datas.append(data)
        
        # 批量处理
        texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) 
                 for msg in batch_messages]
        
        inputs = processor(
            text=texts,
            images=batch_images,
            padding=True,
            return_tensors="pt",
        ).to(model.device)
        
        # 批量生成
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=16,  # ID列表不需要太长，16足够输出 "2, 3, 4"
                do_sample=False,
                use_cache=True,  # 启用KV缓存加速
                num_beams=1,  # 贪婪解码（最快）
            )
        
        # 处理批次结果
        for j in range(len(batch)):
            sample = batch_samples[j]
            data = batch_datas[j]
            
            generated_ids = output_ids[j][len(inputs.input_ids[j]):]
            response = processor.decode(generated_ids, skip_special_tokens=True)
            
            # 解析输出和真实答案
            pred_ids = parse_model_output(response)
            gt_ids = data['answer'].get('target_ids', [data['answer'].get('target_id')])
            
            # 处理gt_ids可能不是列表的情况
            if not isinstance(gt_ids, list):
                gt_ids = [gt_ids] if gt_ids else []
            
            # 计算指标
            metrics = compute_metrics(pred_ids, gt_ids)
            all_metrics.append(metrics)
            
            sample_results.append({
                'sample': sample['json_path'],
                'response': response,
                'pred_ids': pred_ids,
                'gt_ids': gt_ids,
                'exact_match': metrics['exact_match'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1'],
                'main_element': data.get('main_element', {}),
                'candidate_elements': data.get('candidate_elements', []),
            })
            
            # 打印前5个样本的详细输出
            processed_samples += 1
            if processed_samples <= 5:
                print(f"\n{'='*80}")
                print(f"实时输出 - 样本 {processed_samples}: {Path(sample['json_path']).name}")
                print(f"{'='*80}")
                print(f"模型原始输出: '{response}'")
                print(f"解析后的预测ID: {pred_ids}")
                print(f"真实标签ID: {gt_ids}")
                print(f"TP={metrics['true_positive']}, FP={metrics['false_positive']}, FN={metrics['false_negative']}")
                print(f"P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, F1={metrics['f1']:.4f}")
                print(f"{'='*80}")
    
    # 诊断统计
    print("\n" + "="*80)
    print("诊断统计")
    print("="*80)
    empty_predictions = sum(1 for r in sample_results if len(r['pred_ids']) == 0)
    empty_gt = sum(1 for r in sample_results if len(r['gt_ids']) == 0)
    print(f"总样本数: {len(sample_results)}")
    print(f"模型预测为空的样本数: {empty_predictions} ({empty_predictions/len(sample_results)*100:.1f}%)")
    print(f"真实标签为空的样本数: {empty_gt} ({empty_gt/len(sample_results)*100:.1f}%)")
    print(f"模型输出中包含'无'的样本数: {sum(1 for r in sample_results if r['response'].strip() == '无')}")
    
    # 统计最常见的模型输出（前10个）
    from collections import Counter
    response_counter = Counter(r['response'].strip() for r in sample_results)
    print(f"\n最常见的模型输出（前10个）:")
    for response, count in response_counter.most_common(10):
        print(f"  '{response}': {count}次 ({count/len(sample_results)*100:.1f}%)")
    print("="*80 + "\n")
    
    # 汇总结果
    if all_metrics:
        avg_precision = sum(m['precision'] for m in all_metrics) / len(all_metrics)
        avg_recall = sum(m['recall'] for m in all_metrics) / len(all_metrics)
        avg_f1 = sum(m['f1'] for m in all_metrics) / len(all_metrics)
        exact_match_rate = sum(m['exact_match'] for m in all_metrics) / len(all_metrics)
        
        total_tp = sum(m['true_positive'] for m in all_metrics)
        total_fp = sum(m['false_positive'] for m in all_metrics)
        total_fn = sum(m['false_negative'] for m in all_metrics)
        
        micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
        
        results = {
            'num_samples': len(all_metrics),
            'exact_match_rate': exact_match_rate,
            'macro_avg': {'precision': avg_precision, 'recall': avg_recall, 'f1': avg_f1},
            'micro_avg': {'precision': micro_precision, 'recall': micro_recall, 'f1': micro_f1},
            'total': {'true_positive': total_tp, 'false_positive': total_fp, 'false_negative': total_fn},
            'sample_results': sample_results,
        }
        
        # 打印结果
        print("\n" + "="*60)
        print("评估结果")
        print("="*60)
        print(f"测试样本数: {len(all_metrics)}")
        print(f"\n完全匹配率 (Exact Match): {exact_match_rate:.4f} ({exact_match_rate*100:.2f}%)")
        print(f"\n宏平均 (Macro Average):")
        print(f"  精确率 (Precision): {avg_precision:.4f} ({avg_precision*100:.2f}%)")
        print(f"  召回率 (Recall):    {avg_recall:.4f} ({avg_recall*100:.2f}%)")
        print(f"  F1 分数:            {avg_f1:.4f} ({avg_f1*100:.2f}%)")
        print(f"\n微平均 (Micro Average):")
        print(f"  精确率 (Precision): {micro_precision:.4f} ({micro_precision*100:.2f}%)")
        print(f"  召回率 (Recall):    {micro_recall:.4f} ({micro_recall*100:.2f}%)")
        print(f"  F1 分数:            {micro_f1:.4f} ({micro_f1*100:.2f}%)")
        print(f"\n总计:")
        print(f"  正确预测 (TP): {total_tp}")
        print(f"  错误预测 (FP): {total_fp}")
        print(f"  漏检 (FN):     {total_fn}")
        
        # 保存结果
        results_file = output_dir / evaluation_results_name
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n详细结果已保存至: {results_file}")
        
        # 保存详细的模型输出日志（用于调试）
        detailed_log_file = output_dir / 'model_outputs_detailed.jsonl'
        with open(detailed_log_file, 'w', encoding='utf-8') as f:
            for result in sample_results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        print(f"详细输出日志已保存至: {detailed_log_file}")
        
        # 保存前20个样本的可读版本
        readable_log_file = output_dir / 'model_outputs_readable.txt'
        with open(readable_log_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("模型输出详细分析（前20个样本）\n")
            f.write("="*80 + "\n\n")
            
            for idx, result in enumerate(sample_results[:20], 1):
                f.write(f"\n{'='*80}\n")
                f.write(f"样本 {idx}: {Path(result['sample']).name}\n")
                f.write(f"{'='*80}\n")
                
                f.write(f"\n主体元件: {result.get('main_element', {})}\n")
                f.write(f"候选元件数量: {len(result.get('candidate_elements', []))}\n")
                
                f.write(f"\n模型原始输出: '{result['response']}'\n")
                f.write(f"解析后预测ID: {result['pred_ids']}\n")
                f.write(f"真实标签ID: {result['gt_ids']}\n")
                
                f.write(f"\n指标:\n")
                f.write(f"  - Exact Match: {result['exact_match']}\n")
                f.write(f"  - Precision: {result['precision']:.4f}\n")
                f.write(f"  - Recall: {result['recall']:.4f}\n")
                f.write(f"  - F1: {result['f1']:.4f}\n")
        
        print(f"可读输出日志已保存至: {readable_log_file}")
        
        # 生成图表
        generate_evaluation_charts(results, output_dir, paths_cfg)
        
        return results
    
    return None


def generate_evaluation_charts(results: dict, output_dir: Path, paths_cfg: dict = None):
    """生成评估图表"""
    # 使用英文标签避免中文乱码
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False
    
    # 获取文件名配置
    if paths_cfg is None:
        paths_cfg = {}
    evaluation_metrics_name = paths_cfg.get('evaluation_metrics', 'evaluation_metrics.png')
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Chart 1: Metrics Bar Chart
    metrics = ['Precision', 'Recall', 'F1 Score']
    macro_values = [results['macro_avg']['precision'], results['macro_avg']['recall'], results['macro_avg']['f1']]
    micro_values = [results['micro_avg']['precision'], results['micro_avg']['recall'], results['micro_avg']['f1']]
    
    x = range(len(metrics))
    width = 0.35
    
    bars1 = axes[0].bar([i - width/2 for i in x], macro_values, width, label='Macro Avg', color='steelblue', alpha=0.8)
    bars2 = axes[0].bar([i + width/2 for i in x], micro_values, width, label='Micro Avg', color='coral', alpha=0.8)
    
    axes[0].set_ylabel('Score', fontsize=12)
    axes[0].set_title('Model Evaluation Metrics', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics, fontsize=11)
    axes[0].set_ylim(0, 1)
    axes[0].legend(fontsize=10)
    axes[0].grid(axis='y', alpha=0.3)
    
    for bar in bars1:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.02, f'{height:.2%}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.02, f'{height:.2%}', ha='center', va='bottom', fontsize=9)
    
    # Chart 2: Prediction Distribution Pie Chart
    total = results['total']
    labels = ['True Positive\n(TP)', 'False Positive\n(FP)', 'False Negative\n(FN)']
    sizes = [total['true_positive'], total['false_positive'], total['false_negative']]
    colors = ['#4CAF50', '#FF5722', '#FFC107']
    
    if sum(sizes) > 0:
        axes[1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10})
        axes[1].set_title('Prediction Distribution', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    chart_path = output_dir / evaluation_metrics_name
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    print(f"Evaluation chart saved to: {chart_path}")
    plt.close()


def quick_evaluate(model_path: str, data_dir: str, output_dir: str = 'output', config: dict = None, use_validation_set: bool = False):
    """快速评估接口"""
    return evaluate_model(model_path, data_dir, output_dir, config=config, use_validation_set=use_validation_set)


if __name__ == '__main__':
    # 加载配置
    config = load_config()
    
    # 优先使用代码中设置的路径，否则从配置文件读取
    model_path = MODEL_PATH if MODEL_PATH is not None else config['paths'].get('output_dir', 'output') + '/' + config['paths'].get('final_model_subdir', 'final')
    data_dir = PROCESSED_DATA_DIR if PROCESSED_DATA_DIR is not None else config['paths'].get('processed_data_dir', 'processed_data')
    output_dir = OUTPUT_DIR if OUTPUT_DIR is not None else config['paths'].get('output_dir', 'output')
    base_model_path = BASE_MODEL_PATH if BASE_MODEL_PATH is not None else config.get('model', {}).get('name', '/home/ubuntu/model/Qwen2.5-VL-3B-Instruct')
    
    print("="*60)
    print("模型评估配置")
    print("="*60)
    print(f"微调模型路径: {model_path}")
    if MODEL_PATH is not None:
        print("  → 使用代码中指定的路径")
    else:
        print("  → 使用配置文件中的路径")
    
    print(f"\n预处理数据目录: {data_dir}")
    if PROCESSED_DATA_DIR is not None:
        print("  → 使用代码中指定的路径")
    else:
        print("  → 使用配置文件中的路径")
    
    print(f"\n基础模型路径: {base_model_path}")
    if BASE_MODEL_PATH is not None:
        print("  → 使用代码中指定的路径")
    else:
        print("  → 使用配置文件中的路径")
    
    print(f"\n评估输出目录: {output_dir}")
    if OUTPUT_DIR is not None:
        print("  → 使用代码中指定的路径")
    else:
        print("  → 使用配置文件中的路径")
    print("  → 所有评估结果文件将保存到此目录")
    
    print(f"\n批处理配置:")
    print(f"  Batch Size: {EVAL_BATCH_SIZE}")
    print(f"  Max New Tokens: 16")
    print(f"  量化: 无 (使用FP16)")
    print(f"  优化: KV缓存 + CuDNN自动调优")
    print("="*60)
    print()
    
    evaluate_model(model_path, data_dir, output_dir, config=config)
