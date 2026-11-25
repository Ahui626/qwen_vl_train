"""
导出训练数据查看脚本
简化格式：每条关联关系一个样本
"""
import json
from pathlib import Path
from tqdm import tqdm

from dataset import build_prompt, build_answer


def export_training_data(data_dir: str, output_file: str, max_samples: int = 20):
    """导出训练数据为可读格式"""
    data_dir = Path(data_dir)
    
    print(f"扫描数据目录: {data_dir}")
    all_json_files = []
    
    for subfolder in data_dir.iterdir():
        if subfolder.is_dir():
            # 支持多种文件命名格式
            for pattern in ['sample_*.json', 'valve_*.json', 'device_*.json']:
                for json_file in subfolder.glob(pattern):
                    all_json_files.append(json_file)
    
    all_json_files.sort()
    print(f"找到 {len(all_json_files)} 个样本")
    
    if max_samples:
        all_json_files = all_json_files[:max_samples]
        print(f"导出前 {max_samples} 个")
    
    # 导出
    output_path = Path(output_file)
    
    with open(output_path, 'w', encoding='utf-8') as out:
        out.write("="*100 + "\n")
        out.write("训练数据查看（简化格式）\n")
        out.write("="*100 + "\n\n")
        out.write("特点：\n")
        out.write("  - 每条关联关系 = 一个训练样本\n")
        out.write("  - 红色框(ID 0,1) = 主体元件\n")
        out.write("  - 黄色框(ID 2,3,...) = 其他元件\n")
        out.write("  - ID使用简单数字，不用复杂原始ID\n")
        out.write("="*100 + "\n\n")
        
        for idx, json_path in enumerate(tqdm(all_json_files, desc="处理中"), 1):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except:
                continue
            
            image_path = json_path.parent / data['image']
            
            # 构建提示词和答案
            prompt = build_prompt(data['main_element'], data['candidate_elements'])
            answer = build_answer(data['answer'])
            
            out.write(f"\n{'='*100}\n")
            out.write(f"样本 #{idx}\n")
            out.write(f"{'='*100}\n")
            out.write(f"图片: {image_path}\n")
            out.write(f"来源: {data.get('source_folder', 'N/A')}\n")
            
            out.write(f"\n【红色框 - 主体元件】\n")
            main = data['main_element']
            out.write(f"  ID={main['simple_id']}: {main['category']}\n")
            
            out.write(f"\n【黄色框 - 候选元件】\n")
            target_ids = data['answer'].get('target_ids', [data['answer'].get('target_id')])
            for elem in data['candidate_elements']:
                marker = "★" if elem['simple_id'] in target_ids else " "
                out.write(f"  {marker}ID={elem['simple_id']}: {elem['category']}\n")
            
            out.write(f"\n【正确答案】（{len(target_ids)}个关联）\n")
            target_str = ", ".join(str(t) for t in target_ids)
            out.write(f"  红色框({main['simple_id']}) 与 黄色框({target_str}) 有关联\n")
            
            out.write(f"\n{'-'*100}\n")
            out.write("【训练输入 - Prompt】\n")
            out.write(f"{'-'*100}\n")
            out.write(prompt)
            out.write("\n")
            
            out.write(f"\n{'-'*100}\n")
            out.write("【训练输出 - Answer】\n")
            out.write(f"{'-'*100}\n")
            out.write(answer)
            out.write("\n")
            
            out.write("\n")
    
    print(f"\n导出完成: {output_path}")
    
    # 打印第一条示例
    if all_json_files:
        with open(all_json_files[0], 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        prompt = build_prompt(data['main_element'], data['candidate_elements'])
        answer = build_answer(data['answer'])
        
        print("\n" + "="*80)
        print("第一条训练样本示例")
        print("="*80)
        
        print("\n【红色框 - 主体元件】")
        print(f"  ID={data['main_element']['simple_id']}: {data['main_element']['category']}")
        
        print("\n【黄色框 - 候选元件】")
        target_ids = data['answer'].get('target_ids', [data['answer'].get('target_id')])
        for elem in data['candidate_elements']:
            marker = "★" if elem['simple_id'] in target_ids else " "
            print(f"  {marker}ID={elem['simple_id']}: {elem['category']}")
        
        print(f"\n【正确答案】（{len(target_ids)}个关联）")
        print(f"  target_ids={target_ids}")
        
        print(f"\n【Prompt】")
        print(prompt)
        
        print(f"\n【Answer】")
        print(answer)
        print("="*80)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='processed_data_test')
    parser.add_argument('--output', type=str, default='training_data.txt')
    parser.add_argument('--max_samples', type=int, default=20)
    args = parser.parse_args()
    
    export_training_data(args.data_dir, args.output, args.max_samples)

