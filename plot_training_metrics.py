#!/usr/bin/env python3
"""
Training Metrics Visualization Script
Read training logs and plot loss, precision, recall, f1 curves
"""

import json
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
import argparse

# Use default font
matplotlib.rcParams['font.family'] = 'DejaVu Sans'


def load_training_logs(log_file: str) -> dict:
    """Load training logs"""
    # 使用字典去重，确保每个step只保留最后一条记录
    train_data = {}  # {step: loss}
    eval_data = {}   # {step: {loss, precision, recall, f1}}
    
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                step = entry.get('step', None)
                
                if step is None:
                    continue
                
                # Training loss
                if 'loss' in entry and 'eval_loss' not in entry:
                    train_data[step] = entry['loss']
                
                # Evaluation metrics（同一个step可能有多条，保留最后一条）
                if 'eval_loss' in entry:
                    eval_data[step] = {
                        'loss': entry['eval_loss'],
                        'precision': entry.get('eval_precision', 0),
                        'recall': entry.get('eval_recall', 0),
                        'f1': entry.get('eval_f1', 0),
                    }
            except:
                continue
    
    # 转换为列表格式（按step排序）
    logs = {
        'train_steps': [],
        'train_loss': [],
        'eval_steps': [],
        'eval_loss': [],
        'eval_precision': [],
        'eval_recall': [],
        'eval_f1': [],
    }
    
    # 训练数据
    for step in sorted(train_data.keys()):
        logs['train_steps'].append(step)
        logs['train_loss'].append(train_data[step])
    
    # 评估数据
    for step in sorted(eval_data.keys()):
        logs['eval_steps'].append(step)
        logs['eval_loss'].append(eval_data[step]['loss'])
        logs['eval_precision'].append(eval_data[step]['precision'])
        logs['eval_recall'].append(eval_data[step]['recall'])
        logs['eval_f1'].append(eval_data[step]['f1'])
    
    return logs


def plot_metrics(logs: dict, output_file: str):
    """Plot training metrics curves"""
    # 根据是否有指标数据决定布局
    has_metrics = logs['eval_precision'] or logs['eval_recall'] or logs['eval_f1']
    
    if has_metrics:
        # 完整布局：2x2
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Training Metrics', fontsize=18, fontweight='bold')
        
        # 1. Training Loss
        ax1 = axes[0, 0]
        if logs['train_loss']:
            ax1.plot(logs['train_steps'], logs['train_loss'], 
                    label='Train Loss', color='#1f77b4', linewidth=2, alpha=0.8)
        ax1.set_xlabel('Training Steps', fontsize=13)
        ax1.set_ylabel('Loss', fontsize=13)
        ax1.set_title('Training Loss', fontsize=15, fontweight='bold')
        ax1.legend(fontsize=11, loc='upper right')
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        # 2. Evaluation Loss
        ax2 = axes[0, 1]
        if logs['eval_loss']:
            ax2.plot(logs['eval_steps'], logs['eval_loss'], 
                    label='Eval Loss', color='#ff7f0e', linewidth=2.5, 
                    alpha=0.8, marker='o', markersize=5)
        ax2.set_xlabel('Training Steps', fontsize=13)
        ax2.set_ylabel('Loss', fontsize=13)
        ax2.set_title('Evaluation Loss', fontsize=15, fontweight='bold')
        ax2.legend(fontsize=11, loc='upper right')
        ax2.grid(True, alpha=0.3, linestyle='--')
        
        # 3. Precision & Recall
        ax3 = axes[1, 0]
        if logs['eval_precision']:
            ax3.plot(logs['eval_steps'], logs['eval_precision'], 
                    label='Precision', color='#2ca02c', linewidth=2.5, 
                    alpha=0.8, marker='s', markersize=5)
        if logs['eval_recall']:
            ax3.plot(logs['eval_steps'], logs['eval_recall'], 
                    label='Recall', color='#d62728', linewidth=2.5, 
                    alpha=0.8, marker='^', markersize=5)
        ax3.set_xlabel('Training Steps', fontsize=13)
        ax3.set_ylabel('Score', fontsize=13)
        ax3.set_title('Precision & Recall', fontsize=15, fontweight='bold')
        ax3.legend(fontsize=11, loc='lower right')
        ax3.grid(True, alpha=0.3, linestyle='--')
        ax3.set_ylim([0, 1.05])
        
        # 4. F1 Score
        ax4 = axes[1, 1]
        if logs['eval_f1']:
            ax4.plot(logs['eval_steps'], logs['eval_f1'], 
                    label='F1 Score', color='#9467bd', linewidth=2.5, 
                    alpha=0.8, marker='D', markersize=5)
        ax4.set_xlabel('Training Steps', fontsize=13)
        ax4.set_ylabel('F1 Score', fontsize=13)
        ax4.set_title('F1 Score', fontsize=15, fontweight='bold')
        ax4.legend(fontsize=11, loc='lower right')
        ax4.grid(True, alpha=0.3, linestyle='--')
        ax4.set_ylim([0, 1.05])
    else:
        # 简化布局：只有Loss曲线
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Training Metrics', fontsize=18, fontweight='bold')
        
        # 1. Training Loss
        ax1 = axes[0]
        if logs['train_loss']:
            ax1.plot(logs['train_steps'], logs['train_loss'], 
                    label='Train Loss', color='#1f77b4', linewidth=2, alpha=0.8)
        ax1.set_xlabel('Training Steps', fontsize=13)
        ax1.set_ylabel('Loss', fontsize=13)
        ax1.set_title('Training Loss', fontsize=15, fontweight='bold')
        ax1.legend(fontsize=11, loc='upper right')
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        # 2. Evaluation Loss
        ax2 = axes[1]
        if logs['eval_loss']:
            ax2.plot(logs['eval_steps'], logs['eval_loss'], 
                    label='Eval Loss', color='#ff7f0e', linewidth=2.5, 
                    alpha=0.8, marker='o', markersize=5)
        ax2.set_xlabel('Training Steps', fontsize=13)
        ax2.set_ylabel('Loss', fontsize=13)
        ax2.set_title('Evaluation Loss', fontsize=15, fontweight='bold')
        ax2.legend(fontsize=11, loc='upper right')
        ax2.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Training curves saved to: {output_file}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot training metrics curves')
    parser.add_argument('--log_file', type=str, default='output/training_log.jsonl',
                      help='Training log file path')
    parser.add_argument('--output', type=str, default='output/training_curves.png',
                      help='Output image path')
    args = parser.parse_args()
    
    log_file = Path(args.log_file)
    if not log_file.exists():
        print(f"Error: Log file not found: {log_file}")
        return
    
    print(f"Reading training logs: {log_file}")
    logs = load_training_logs(str(log_file))
    
    if not logs['train_steps'] and not logs['eval_steps']:
        print("Error: No valid training data found in log file")
        return
    
    print(f"Found {len(logs['train_loss'])} training records")
    print(f"Found {len(logs['eval_loss'])} evaluation records")
    
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    print("Plotting training curves...")
    plot_metrics(logs, str(output_file))
    
    print("\n✅ Done!")


if __name__ == '__main__':
    main()

