"""
训练可视化工具
"""
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
from pathlib import Path
import pandas as pd


def plot_training_curves(log_file: str, output_dir: str, config: dict = None):
    """
    绘制训练曲线
    从训练日志中提取 loss、learning_rate 等指标并绘图
    """
    output_dir = Path(output_dir)
    
    # 获取文件名配置
    if config and 'paths' in config:
        training_curves_name = config['paths']['training_curves']
        loss_comparison_name = config['paths']['loss_comparison']
    else:
        # 默认文件名（向后兼容）
        training_curves_name = 'training_curves.png'
        loss_comparison_name = 'loss_comparison.png'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 读取日志文件
    logs = []
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                log_entry = json.loads(line.strip())
                logs.append(log_entry)
            except json.JSONDecodeError:
                continue
    
    if not logs:
        print("No valid log entries found")
        return
    
    df = pd.DataFrame(logs)
    
    # 使用英文标签避免中文乱码
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Process Visualization', fontsize=16, fontweight='bold')
    
    # 1. Training Loss Curve
    if 'loss' in df.columns:
        train_loss = df[df['loss'].notna()][['step', 'loss']]
        if not train_loss.empty:
            axes[0, 0].plot(train_loss['step'], train_loss['loss'], 'b-', linewidth=2, label='Train Loss')
            axes[0, 0].set_xlabel('Training Steps', fontsize=12)
            axes[0, 0].set_ylabel('Loss', fontsize=12)
            axes[0, 0].set_title('Training Loss Curve', fontsize=14, fontweight='bold')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].legend()
    
    # 2. Validation Loss Curve
    if 'eval_loss' in df.columns:
        eval_loss = df[df['eval_loss'].notna()][['step', 'eval_loss']]
        if not eval_loss.empty:
            axes[0, 1].plot(eval_loss['step'], eval_loss['eval_loss'], 'r-', linewidth=2, label='Eval Loss')
            axes[0, 1].set_xlabel('Training Steps', fontsize=12)
            axes[0, 1].set_ylabel('Loss', fontsize=12)
            axes[0, 1].set_title('Validation Loss Curve', fontsize=14, fontweight='bold')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].legend()
    
    # 3. Learning Rate Curve
    if 'learning_rate' in df.columns:
        lr_data = df[df['learning_rate'].notna()][['step', 'learning_rate']]
        if not lr_data.empty:
            axes[1, 0].plot(lr_data['step'], lr_data['learning_rate'], 'g-', linewidth=2, label='Learning Rate')
            axes[1, 0].set_xlabel('Training Steps', fontsize=12)
            axes[1, 0].set_ylabel('Learning Rate', fontsize=12)
            axes[1, 0].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].legend()
            axes[1, 0].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # 4. Gradient Norm
    if 'grad_norm' in df.columns:
        grad_data = df[df['grad_norm'].notna()][['step', 'grad_norm']]
        if not grad_data.empty:
            axes[1, 1].plot(grad_data['step'], grad_data['grad_norm'], 'm-', linewidth=2, label='Gradient Norm')
            axes[1, 1].set_xlabel('Training Steps', fontsize=12)
            axes[1, 1].set_ylabel('Gradient Norm', fontsize=12)
            axes[1, 1].set_title('Gradient Norm Changes', fontsize=14, fontweight='bold')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].legend()
    
    plt.tight_layout()
    
    # 保存图表
    output_path = output_dir / training_curves_name
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to: {output_path}")
    plt.close()
    
    # 绘制训练和验证损失对比图
    if 'loss' in df.columns and 'eval_loss' in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        train_loss = df[df['loss'].notna()][['step', 'loss']]
        eval_loss = df[df['eval_loss'].notna()][['step', 'eval_loss']]
        
        if not train_loss.empty:
            ax.plot(train_loss['step'], train_loss['loss'], 'b-', linewidth=2, label='Train Loss', alpha=0.7)
        if not eval_loss.empty:
            ax.plot(eval_loss['step'], eval_loss['eval_loss'], 'r-', linewidth=2, label='Eval Loss', alpha=0.7)
        
        ax.set_xlabel('Training Steps', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Train vs Eval Loss Comparison', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=11)
        
        output_path = output_dir / loss_comparison_name
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Loss comparison chart saved to: {output_path}")
        plt.close()


def generate_training_report(log_file: str, output_dir: str, config: dict):
    """生成训练报告"""
    output_dir = Path(output_dir)
    
    # 获取文件名配置
    if 'paths' in config:
        report_name = config['paths']['training_report']
    else:
        report_name = 'training_report.txt'
    
    # 读取日志
    logs = []
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                logs.append(json.loads(line.strip()))
            except:
                continue
    
    if not logs:
        return
    
    df = pd.DataFrame(logs)
    
    # 生成报告
    report = []
    report.append("=" * 60)
    report.append("训练报告")
    report.append("=" * 60)
    report.append("")
    
    # 配置信息
    report.append("## 训练配置")
    report.append(f"- 模型: {config['model']['name']}")
    report.append(f"- 训练轮数: {config['training']['num_epochs']}")
    report.append(f"- Batch Size: {config['training']['batch_size']}")
    report.append(f"- 梯度累积步数: {config['training']['gradient_accumulation_steps']}")
    report.append(f"- 学习率: {config['training']['learning_rate']}")
    report.append(f"- LoRA r: {config['lora']['r']}")
    report.append(f"- LoRA alpha: {config['lora']['alpha']}")
    report.append("")
    
    # 训练统计
    report.append("## 训练统计")
    if 'loss' in df.columns:
        train_losses = df[df['loss'].notna()]['loss']
        if not train_losses.empty:
            report.append(f"- 初始训练损失: {train_losses.iloc[0]:.4f}")
            report.append(f"- 最终训练损失: {train_losses.iloc[-1]:.4f}")
            report.append(f"- 最低训练损失: {train_losses.min():.4f}")
            report.append(f"- 平均训练损失: {train_losses.mean():.4f}")
    
    if 'eval_loss' in df.columns:
        eval_losses = df[df['eval_loss'].notna()]['eval_loss']
        if not eval_losses.empty:
            report.append(f"- 最低验证损失: {eval_losses.min():.4f}")
            report.append(f"- 最终验证损失: {eval_losses.iloc[-1]:.4f}")
    
    if 'train_runtime' in df.columns:
        runtime = df[df['train_runtime'].notna()]['train_runtime'].iloc[-1]
        report.append(f"- 训练总时长: {runtime:.2f} 秒 ({runtime/60:.2f} 分钟)")
    
    if 'train_samples_per_second' in df.columns:
        speed = df[df['train_samples_per_second'].notna()]['train_samples_per_second'].iloc[-1]
        report.append(f"- 训练速度: {speed:.2f} 样本/秒")
    
    report.append("")
    report.append("=" * 60)
    
    # 保存报告
    report_path = output_dir / report_name
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f"训练报告已保存至: {report_path}")
    
    # 打印到控制台
    print('\n'.join(report))


if __name__ == '__main__':
    import sys
    if len(sys.argv) >= 2:
        log_file = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) >= 3 else 'output'
        plot_training_curves(log_file, output_dir)

