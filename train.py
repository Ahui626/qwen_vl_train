"""
Qwen2.5-VL 关联关系训练脚本
"""
import os
import sys
import json
import yaml
import argparse
from pathlib import Path


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def setup_gpu(config: dict):
    """设置 GPU - 必须在导入 torch 之前调用"""
    # 如果环境变量已经设置（多卡训练），则优先使用环境变量
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        visible = config['gpu']['visible_devices']
        os.environ['CUDA_VISIBLE_DEVICES'] = visible
        print(f"Using GPU from config: {visible}")
    else:
        print(f"Using GPU from environment: {os.environ['CUDA_VISIBLE_DEVICES']}")


# 主程序入口 - 在导入 torch 之前设置 GPU
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='训练 Qwen2.5-VL 关联关系模型')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, 
                        help='从指定checkpoint继续训练 (例如: output/checkpoint-1500)')
    args = parser.parse_args()
    
    # 在导入 torch 之前设置 GPU
    config = load_config(args.config)
    setup_gpu(config)
    
    # 现在才导入 torch 和其他库
    import torch
    from transformers import (
        AutoModelForVision2Seq,
        AutoProcessor,
        TrainingArguments,
        Trainer,
        TrainerCallback,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    
    from dataset import create_datasets
    from evaluate import quick_evaluate
    
    
    class EvaluationCallback(TrainerCallback):
        """自定义回调：计算和记录评估指标"""
        def __init__(self, eval_dataset, processor, log_file, test_samples_dir=None):
            self.eval_dataset = eval_dataset
            self.processor = processor
            self.log_file = log_file
            self.log_handle = open(log_file, 'w', encoding='utf-8')
            self.metrics_history = []
            
            # 如果提供了测试集目录，使用测试集；否则使用验证集
            if test_samples_dir and Path(test_samples_dir).exists():
                # 加载测试集样本
                test_dir = Path(test_samples_dir)
                self.test_samples = []
                for subfolder in sorted(test_dir.iterdir()):
                    if subfolder.is_dir():
                        for pattern in ['sample_*.json', 'valve_*.json', 'device_*.json', 'tag_*.json']:
                            for json_file in sorted(subfolder.glob(pattern)):
                                self.test_samples.append(str(json_file))
                
                # 限制测试样本数量（避免评估太慢）
                max_eval_samples = 50
                if len(self.test_samples) > max_eval_samples:
                    import random
                    random.seed(42)
                    self.test_samples = random.sample(self.test_samples, max_eval_samples)
                else:
                    self.test_samples = self.test_samples[:max_eval_samples]
                
                print(f"评估回调：使用独立测试集，共 {len(self.test_samples)} 个样本")
                self.use_test_set = True
            else:
                # 使用验证集的所有样本（因为验证集已经在外部被限制为50个）
                # 注意：如果eval_dataset是Subset，len会返回子集大小
                self.eval_sample_indices = list(range(len(eval_dataset)))
                print(f"评估回调：使用验证集，共 {len(self.eval_sample_indices)} 个样本")
                self.use_test_set = False
        
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                # 过滤掉评估日志（评估指标由 on_evaluate 专门处理）
                # 只记录训练loss，避免重复写入导致曲线出现三角形
                if 'eval_loss' in logs:
                    return  # 跳过评估日志，由 on_evaluate 统一处理
                
                logs['step'] = state.global_step
                self.log_handle.write(json.dumps(logs, ensure_ascii=False) + '\n')
                self.log_handle.flush()
        
        def on_evaluate(self, args, state, control, model, metrics=None, **kwargs):
            """在每次评估时计算 precision、recall、f1"""
            if metrics is None:
                return
            
            # 在验证集上进行推理并计算指标
            # 策略：重新使用 processor 处理原始数据，确保格式正确
            try:
                import torch
                import json
                from pathlib import Path
                from PIL import Image
                from dataset import resize_image
                
                model.eval()
                
                # 使用固定的评估样本集（避免每次评估结果波动）
                all_precisions = []
                all_recalls = []
                all_f1s = []
                
                # 根据数据源选择样本
                if self.use_test_set:
                    # 使用测试集
                    sample_paths = self.test_samples
                else:
                    # 使用验证集
                    sample_paths = [self.eval_dataset.samples[i] for i in self.eval_sample_indices]
                
                for json_path_str in sample_paths:
                    # 从文件路径加载数据
                    json_path = Path(json_path_str)
                    
                    # 重新加载原始数据
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # 加载图片
                    image_path = json_path.parent / data['image']
                    image = Image.open(image_path).convert('RGB')
                    image = resize_image(image)
                    
                    # 构建问题（不需要答案）
                    from dataset import build_prompt
                    question = build_prompt(data['main_element'], data['candidate_elements'])
                    
                    # 使用 processor 处理（确保格式正确）
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": question},
                            ],
                        },
                    ]
                    
                    text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    inputs = self.processor(
                        text=[text],
                        images=[image],
                        padding=True,
                        return_tensors="pt",
                    ).to(model.device)
                    
                    # 生成预测
                    with torch.no_grad():
                        output_ids = model.generate(
                            **inputs,
                            max_new_tokens=16,
                            do_sample=False,
                        )
                    
                    # 解码（注意：这里的inputs已经是batch格式）
                    input_len = inputs['input_ids'].shape[1]
                    generated_ids = output_ids[0][input_len:]
                    response = self.processor.decode(generated_ids, skip_special_tokens=True)
                    
                    # 解析预测
                    pred_ids = self.parse_output(response)
                    
                    # 获取真实标签 - 从JSON数据中获取
                    gt_ids = []
                    answer_dict = data.get('answer', {})
                    # answer是字典，包含 target_ids 或 target_id
                    if isinstance(answer_dict, dict):
                        # 优先使用 target_ids（多个关联的情况）
                        if 'target_ids' in answer_dict:
                            target_ids = answer_dict['target_ids']
                            if isinstance(target_ids, list):
                                gt_ids = [int(x) for x in target_ids if isinstance(x, (int, str)) and str(x).isdigit()]
                            else:
                                gt_ids = [int(target_ids)] if str(target_ids).isdigit() else []
                        # 兼容旧格式 target_id（单个关联）
                        elif 'target_id' in answer_dict:
                            target_id = answer_dict['target_id']
                            gt_ids = [int(target_id)] if str(target_id).isdigit() else []
                    else:
                        # 如果answer本身就是列表或单个ID（兼容性处理）
                        if isinstance(answer_dict, list):
                            gt_ids = [int(x) for x in answer_dict if isinstance(x, (int, str)) and str(x).isdigit()]
                        else:
                            gt_ids = self.parse_output(str(answer_dict))
                    
                    # 计算单样本指标
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
                    
                    all_precisions.append(precision)
                    all_recalls.append(recall)
                    all_f1s.append(f1)
                
                # 计算平均指标
                avg_precision = sum(all_precisions) / len(all_precisions) if all_precisions else 0
                avg_recall = sum(all_recalls) / len(all_recalls) if all_recalls else 0
                avg_f1 = sum(all_f1s) / len(all_f1s) if all_f1s else 0
                
                # 添加到metrics
                metrics['eval_precision'] = avg_precision
                metrics['eval_recall'] = avg_recall
                metrics['eval_f1'] = avg_f1
                metrics['eval_samples'] = len(self.test_samples) if self.use_test_set else len(self.eval_sample_indices)
                
                # 保存指标历史
                metric_entry = {
                    'step': state.global_step,
                    'epoch': state.epoch,
                    'eval_loss': metrics.get('eval_loss', 0),
                    'eval_precision': avg_precision,
                    'eval_recall': avg_recall,
                    'eval_f1': avg_f1,
                }
                self.metrics_history.append(metric_entry)
                
                # 立即写入日志
                self.log_handle.write(json.dumps(metric_entry, ensure_ascii=False) + '\n')
                self.log_handle.flush()
                
            except Exception as e:
                import traceback
                print(f"Warning: Failed to compute custom metrics: {e}")
                traceback.print_exc()
            
            model.train()
        
        def parse_output(self, output: str) -> list:
            """解析模型输出"""
            output = output.strip()
            
            # 处理空输出或"无"/"NONE"的情况
            if not output or output == "无" or output.upper() == "NONE":
                return []
            
            try:
                ids = []
                for part in output.replace(' ', '').split(','):
                    part = part.strip()
                    if part.isdigit():
                        ids.append(int(part))
                return ids
            except:
                pass
            try:
                if output.isdigit():
                    return [int(output)]
            except:
                pass
            return []
        
        def on_train_end(self, args, state, control, **kwargs):
            self.log_handle.close()
            
            # 保存指标历史
            metrics_file = Path(self.log_file).parent / 'metrics_history.json'
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(self.metrics_history, f, ensure_ascii=False, indent=2)
    
    
    def create_model_and_processor(config: dict):
        """加载模型和处理器"""
        model_name = config['model']['name']
        print(f"Loading model: {model_name}")
        
        # 检查是否为本地路径
        model_path = Path(model_name).expanduser().resolve()
        if model_path.exists():
            print(f"Loading from local path: {model_path}")
            model_name = str(model_path)
        
        # 加载处理器
        processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        
        # 检测是否在DDP环境中
        is_ddp = 'LOCAL_RANK' in os.environ or 'RANK' in os.environ
        
        # 加载模型
        if is_ddp:
            # DDP模式：不使用device_map，模型会被DDP包装后分配到各GPU
            print("DDP模式：加载模型到当前进程的GPU")
            model = AutoModelForVision2Seq.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
        else:
            # 单进程模式：检查GPU数量
            num_gpus = torch.cuda.device_count()
            if num_gpus > 1:
                # 多GPU：使用模型并行
                print(f"单进程模式：检测到{num_gpus}个GPU，使用模型并行")
                model = AutoModelForVision2Seq.from_pretrained(
                    model_name,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True,
                )
            else:
                # 单GPU：直接加载
                print(f"单进程模式：使用单GPU训练")
                model = AutoModelForVision2Seq.from_pretrained(
                    model_name,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                )
        
        # 梯度检查点（启用以降低显存峰值）
        if not is_ddp:
            # 非DDP模式（单卡或模型并行），启用梯度检查点降低显存峰值
            model.gradient_checkpointing_enable()
            if num_gpus > 1:
                print(f"模型并行（{num_gpus}卡）：已启用梯度检查点（降低显存峰值）")
            else:
                print("单卡训练：已启用梯度检查点（降低显存峰值）")
        else:
            # DDP模式下不启用梯度检查点，通过降低batch_size来适配显存
            print("DDP模式：禁用梯度检查点（避免DDP冲突）")
        
        # 应用 LoRA
        if config['lora']['enabled']:
            lora_config = LoraConfig(
                r=config['lora']['r'],
                lora_alpha=config['lora']['alpha'],
                lora_dropout=config['lora']['dropout'],
                target_modules=config['lora']['target_modules'],
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
        
        return model, processor
    
    
    def parse_model_output(output: str) -> list:
        """解析模型输出的ID列表
        
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
    
    
    def compute_metrics_for_trainer(eval_pred):
        """计算训练过程中的评估指标"""
        predictions, labels = eval_pred
        
        # predictions 是 logits，需要解码
        # 这里我们需要使用 processor 来解码
        # 但是在 compute_metrics 中无法直接访问 processor
        # 所以我们先返回一个占位符，实际指标通过回调计算
        return {}
    
    
    def collate_fn(batch):
        """数据整理函数"""
        # 找到最大长度
        max_len = max(item['input_ids'].shape[0] for item in batch)
        
        # Padding
        input_ids = []
        attention_mask = []
        pixel_values = []
        image_grid_thw = []
        
        for item in batch:
            seq_len = item['input_ids'].shape[0]
            pad_len = max_len - seq_len
            
            # Padding input_ids
            padded_ids = torch.cat([
                item['input_ids'],
                torch.zeros(pad_len, dtype=item['input_ids'].dtype)
            ])
            input_ids.append(padded_ids)
            
            # Padding attention_mask
            padded_mask = torch.cat([
                item['attention_mask'],
                torch.zeros(pad_len, dtype=item['attention_mask'].dtype)
            ])
            attention_mask.append(padded_mask)
            
            if 'pixel_values' in item:
                pixel_values.append(item['pixel_values'])
            if 'image_grid_thw' in item:
                # 确保 image_grid_thw 是 2D 的 [num_images, 3]
                grid = item['image_grid_thw']
                if grid.dim() == 1:
                    grid = grid.unsqueeze(0)
                image_grid_thw.append(grid)
        
        result = {
            'input_ids': torch.stack(input_ids),
            'attention_mask': torch.stack(attention_mask),
            'labels': torch.stack(input_ids).clone(),  # 使用 input_ids 作为 labels
        }
        
        if pixel_values:
            result['pixel_values'] = torch.cat(pixel_values, dim=0)
        if image_grid_thw:
            # 正确拼接 image_grid_thw，保持 [total_images, 3] 的形状
            result['image_grid_thw'] = torch.cat(image_grid_thw, dim=0)
        
        return result
    
    
    # 主训练流程
    # 检测是否在DDP环境中
    is_ddp = 'LOCAL_RANK' in os.environ or 'RANK' in os.environ
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    # 只在主进程打印
    if local_rank == 0:
        print("=" * 50)
        print("Starting training preparation...")
        print("=" * 50)
    
    # 加载模型
    model, processor = create_model_and_processor(config)
    
    # 创建数据集（使用预处理后的数据）
    paths_cfg = config['paths']
    data_dir = Path(paths_cfg['processed_data_dir'])
    if not data_dir.exists():
        print(f"错误: 预处理数据目录不存在: {data_dir}")
        print("请先运行数据预处理: python preprocess.py --config config.yaml")
        sys.exit(1)
    
    train_dataset, eval_dataset_full = create_datasets(
        str(data_dir),
        processor,
        train_ratio=config['training']['train_ratio']
    )
    
    # 限制验证集大小为50个样本（加快评估速度）
    max_eval_samples = 50
    if len(eval_dataset_full) > max_eval_samples:
        # 创建一个子集
        from torch.utils.data import Subset
        eval_indices = list(range(max_eval_samples))
        eval_dataset = Subset(eval_dataset_full, eval_indices)
        print(f"验证集限制为 {max_eval_samples} 个样本（原始: {len(eval_dataset_full)}）")
    else:
        eval_dataset = eval_dataset_full
        print(f"验证集共 {len(eval_dataset)} 个样本")
    
    # 训练参数
    train_cfg = config['training']
    output_dir = Path(paths_cfg['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 基础训练参数
    training_args_dict = {
        "output_dir": str(output_dir),
        "num_train_epochs": train_cfg['num_epochs'],
        "per_device_train_batch_size": train_cfg['batch_size'],
        "per_device_eval_batch_size": train_cfg['batch_size'],
        "gradient_accumulation_steps": train_cfg['gradient_accumulation_steps'],
        "learning_rate": train_cfg['learning_rate'],
        "warmup_ratio": train_cfg['warmup_ratio'],
        "weight_decay": train_cfg['weight_decay'],
        "max_grad_norm": train_cfg['max_grad_norm'],
        "logging_steps": train_cfg['logging_steps'],
        "save_steps": train_cfg['save_steps'],
        "eval_strategy": "steps",
        "eval_steps": train_cfg['eval_steps'],
        "save_total_limit": train_cfg['save_total_limit'],
        "bf16": True,
        "dataloader_pin_memory": False,
        "remove_unused_columns": False,
        "report_to": "none",
        "logging_first_step": True,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
    }
    
    # 如果在DDP环境，添加DDP配置
    if is_ddp:
        print("检测到DDP环境，启用分布式训练")
        training_args_dict.update({
            "ddp_find_unused_parameters": True,  # LoRA训练需要True，忽略冻结的参数
            "ddp_backend": "nccl",
        })
    else:
        print("单进程模式（使用 torchrun 可启用多卡加速）")
    
    training_args = TrainingArguments(**training_args_dict)
    
    # 创建日志回调
    log_file = output_dir / paths_cfg['training_log']
    
    # 如果配置了测试集，使用测试集进行评估；否则使用验证集
    test_data_dir = paths_cfg.get('test_data_dir', '')
    test_samples_dir = test_data_dir if test_data_dir and test_data_dir.strip() else None
    
    eval_callback = EvaluationCallback(eval_dataset, processor, str(log_file), test_samples_dir=test_samples_dir)
    
    # 创建 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
        callbacks=[eval_callback],
    )
    
    # 开始训练（只在主进程打印）
    if local_rank == 0:
        print("=" * 50)
        print("Starting training...")
        print(f"Train samples: {len(train_dataset)}")
        print(f"Eval samples: {len(eval_dataset)}")
        
        # 检查是否从checkpoint恢复
        if args.resume_from_checkpoint:
            checkpoint_path = Path(args.resume_from_checkpoint)
            if checkpoint_path.exists():
                print(f"从checkpoint恢复训练: {checkpoint_path}")
            else:
                print(f"警告: 指定的checkpoint不存在: {checkpoint_path}")
                print("将从头开始训练")
                args.resume_from_checkpoint = None
        else:
            print("从头开始训练")
        
        print("=" * 50)
    
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    
    # 保存最终模型
    final_path = output_dir / paths_cfg['final_model_subdir']
    trainer.save_model(str(final_path))
    processor.save_pretrained(str(final_path))
    print(f"Model saved to {final_path}")
    
    # 生成可视化图表
    print("\n" + "="*50)
    print("Generating training visualizations...")
    print("="*50)
    try:
        import subprocess
        curve_file = output_dir / 'training_curves.png'
        subprocess.run([
            'python', 'plot_training_metrics.py',
            '--log_file', str(log_file),
            '--output', str(curve_file)
        ], check=True)
        print(f"✓ Training curves saved: {curve_file}")
    except Exception as e:
        print(f"Warning: Failed to generate visualizations: {e}")
        print("You can manually run: python plot_training_metrics.py --log_file output/training_log.jsonl")
    
    # 最终评估模型
    print("\n" + "="*50)
    
    # 确定使用哪个数据集进行评估
    test_data_dir_config = paths_cfg.get('test_data_dir', '')
    if test_data_dir_config and test_data_dir_config.strip():
        # 使用指定的测试集
        eval_data_path = Path(test_data_dir_config)
        eval_set_name = "独立测试集"
        print(f"使用独立测试集评估: {eval_data_path}")
    else:
        # 使用验证集（训练过程中未用于参数更新的10%数据）
        eval_data_path = data_dir
        eval_set_name = "验证集"
        print(f"使用验证集评估（训练数据的10%）")
    
    print("="*50)
    try:
        # 释放训练资源
        del trainer
        del model
        torch.cuda.empty_cache()
        
        # 运行评估
        eval_results = quick_evaluate(
            str(final_path), 
            str(eval_data_path), 
            str(output_dir), 
            config,
            use_validation_set=(not test_data_dir_config or not test_data_dir_config.strip())
        )
    except Exception as e:
        print(f"Warning: Failed to evaluate model: {e}")
        eval_results = None
    
    # 打印最终总结
    print("\n" + "="*60)
    print("🎉 训练完成！")
    print("="*60)
    print("\n📊 生成的文件：")
    print(f"  - 训练曲线图:     {output_dir}/training_curves.png")
    print(f"  - 训练日志:       {output_dir}/{paths_cfg['training_log']}")
    print(f"  - 评估指标图:     {output_dir}/{paths_cfg['evaluation_metrics']}")
    print(f"  - 评估结果JSON:   {output_dir}/{paths_cfg['evaluation_results']}")
    print(f"  - 模型文件:       {output_dir}/{paths_cfg['final_model_subdir']}/")
    
    if eval_results:
        print("\n📈 核心评估指标：")
        print(f"  - 精确率 (Precision): {eval_results['micro_avg']['precision']:.2%}")
        print(f"  - 召回率 (Recall):    {eval_results['micro_avg']['recall']:.2%}")
        print(f"  - F1 分数:            {eval_results['micro_avg']['f1']:.2%}")
    
    print("\n" + "="*60)
