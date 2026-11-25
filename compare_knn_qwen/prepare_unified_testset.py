"""
统一KNN和Qwen的测试集Ground Truth

功能：
1. 从原始测试集读取元件和关联数据
2. 过滤掉仪表相关的关联关系
3. 为带"阀"字的符号元件添加无关联样本（如果它们没有关联）
4. 保存处理后的数据到新目录

这样KNN和Qwen评估就使用相同的ground truth
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Set
from collections import defaultdict


# ======================== 配置 ========================

# 原始测试集目录
SOURCE_TEST_DIR = Path(__file__).parent / "测试集"

# 处理后的测试集目录
OUTPUT_TEST_DIR = Path(__file__).parent / "测试集_统一GT"

# =====================================================


def is_valve_element(category: str, subclass: str) -> bool:
    """
    判断是否为阀门类元件（符号类且小类别名包含"阀"字）
    参考 preprocess.py 的判断逻辑
    """
    return category == '符号' and '阀' in subclass


def load_elements_txt(txt_path: Path) -> List[Dict]:
    """
    加载元件标注txt文件
    格式：id,大类,小类,x1,y1,x2,y2
    
    Returns:
        元件列表
    """
    elements = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split(',')
            if len(parts) < 7:
                print(f"  ⚠️ 跳过格式错误的行: {line[:50]}")
                continue
            
            elem_id = parts[0]
            category = parts[1]
            subclass = parts[2]
            x1, y1, x2, y2 = map(float, parts[3:7])
            
            elements.append({
                'id': elem_id,
                'category': category,
                'subclass': subclass,
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2
            })
    
    return elements


def load_associations_txt(txt_path: Path) -> List[Dict]:
    """
    加载关联关系txt文件
    格式：关联id,类型,主体大类,关联元件大类,主体x1,主体y1,主体x2,主体y2,关联x1,关联y1,关联x2,关联y2,主体元件id,关联元件id
    
    Returns:
        关联关系列表
    """
    associations = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split(',')
            if len(parts) < 14:
                print(f"  ⚠️ 跳过格式错误的关联行: {line[:50]}")
                continue
            
            associations.append({
                'assoc_id': parts[0],
                'relation_type': parts[1],
                'main_category': parts[2],
                'target_category': parts[3],
                'main_x1': float(parts[4]),
                'main_y1': float(parts[5]),
                'main_x2': float(parts[6]),
                'main_y2': float(parts[7]),
                'target_x1': float(parts[8]),
                'target_y1': float(parts[9]),
                'target_x2': float(parts[10]),
                'target_y2': float(parts[11]),
                'main_elem_id': parts[12],
                'target_elem_id': parts[13]
            })
    
    return associations


def save_elements_txt(elements: List[Dict], output_path: Path):
    """保存元件到txt文件"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for elem in elements:
            line = f"{elem['id']},{elem['category']},{elem['subclass']},{elem['x1']},{elem['y1']},{elem['x2']},{elem['y2']}\n"
            f.write(line)


def save_associations_txt(associations: List[Dict], output_path: Path):
    """保存关联关系到txt文件"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for assoc in associations:
            line = (f"{assoc['assoc_id']},{assoc['relation_type']},"
                   f"{assoc['main_category']},{assoc['target_category']},"
                   f"{assoc['main_x1']},{assoc['main_y1']},{assoc['main_x2']},{assoc['main_y2']},"
                   f"{assoc['target_x1']},{assoc['target_y1']},{assoc['target_x2']},{assoc['target_y2']},"
                   f"{assoc['main_elem_id']},{assoc['target_elem_id']}\n")
            f.write(line)


def process_single_folder(source_folder: Path, output_folder: Path) -> Dict:
    """
    处理单个测试文件夹
    
    Returns:
        统计信息字典
    """
    folder_name = source_folder.name
    print(f"\n{'='*60}")
    print(f"处理: {folder_name}")
    print('='*60)
    
    # 1. 查找元件和关联文件
    elements_txt = source_folder / f"{folder_name}.txt"
    assoc_txt = source_folder / f"{folder_name}_关联.txt"
    
    if not elements_txt.exists():
        print(f"❌ 元件文件不存在: {elements_txt.name}")
        return None
    
    if not assoc_txt.exists():
        print(f"❌ 关联文件不存在: {assoc_txt.name}")
        return None
    
    # 2. 加载数据
    print(f"📂 加载元件和关联数据...")
    elements = load_elements_txt(elements_txt)
    associations = load_associations_txt(assoc_txt)
    
    print(f"  ✅ 原始元件数: {len(elements)}")
    print(f"  ✅ 原始关联数: {len(associations)}")
    
    # 3. 创建元件ID到元件的映射
    element_map = {e['id']: e for e in elements}
    
    # 4. 过滤掉仪表相关的关联
    filtered_associations = []
    filtered_count = 0
    
    for assoc in associations:
        # 检查主元件和从元件是否为仪表
        main_elem = element_map.get(assoc['main_elem_id'])
        target_elem = element_map.get(assoc['target_elem_id'])
        
        # 如果任一元件不存在，跳过
        if not main_elem or not target_elem:
            filtered_count += 1
            continue
        
        # 如果任一元件是仪表，过滤掉
        if main_elem['category'] == '仪表' or target_elem['category'] == '仪表':
            filtered_count += 1
            continue
        
        filtered_associations.append(assoc)
    
    print(f"  🗑️  过滤仪表关联: {filtered_count} 条")
    print(f"  ✅ 保留关联数: {len(filtered_associations)}")
    
    # 5. 统计关联中出现的主元件ID
    main_elem_ids_in_assoc = set()
    for assoc in filtered_associations:
        main_elem_ids_in_assoc.add(assoc['main_elem_id'])
    
    # 6. 找到所有带阀字的符号元件
    valve_elements = [e for e in elements if is_valve_element(e['category'], e['subclass'])]
    print(f"  🔧 找到带阀字的符号: {len(valve_elements)} 个")
    
    # 7. 找到没有关联的阀门元件（没有在关联中作为主元件出现）
    unassociated_valves = []
    for valve_elem in valve_elements:
        if valve_elem['id'] not in main_elem_ids_in_assoc:
            unassociated_valves.append(valve_elem)
    
    print(f"  ➕ 无关联的阀门元件: {len(unassociated_valves)} 个")
    
    # 8. 元件列表保持不变（所有元件都保留，包括无关联的阀门）
    # 注意：在Qwen的测试集中，无关联的阀门元件会作为单独的样本，target_ids为空
    
    # 9. 创建输出文件夹
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # 10. 保存处理后的数据
    output_elements_txt = output_folder / f"{folder_name}.txt"
    output_assoc_txt = output_folder / f"{folder_name}_关联.txt"
    
    save_elements_txt(elements, output_elements_txt)
    save_associations_txt(filtered_associations, output_assoc_txt)
    
    print(f"  💾 已保存到: {output_folder.name}")
    
    # 11. 复制图片和JSON文件（如果存在）
    import shutil
    for ext in ['.jpg', '.jpeg', '.png', '.json']:
        source_files = list(source_folder.glob(f"*{ext}"))
        for src_file in source_files:
            dst_file = output_folder / src_file.name
            if not dst_file.exists():
                shutil.copy2(src_file, dst_file)
    
    print(f"  📄 已复制图片和JSON文件")
    
    return {
        'folder_name': folder_name,
        'original_elements': len(elements),
        'original_associations': len(associations),
        'filtered_associations': len(filtered_associations),
        'removed_associations': filtered_count,
        'total_valve_elements': len(valve_elements),
        'unassociated_valves': len(unassociated_valves)
    }


def main():
    """主函数"""
    print("="*60)
    print("统一KNN和Qwen测试集Ground Truth")
    print("="*60)
    print(f"源目录: {SOURCE_TEST_DIR}")
    print(f"输出目录: {OUTPUT_TEST_DIR}")
    print()
    
    # 创建输出目录
    OUTPUT_TEST_DIR.mkdir(parents=True, exist_ok=True)
    
    # 获取所有测试文件夹
    test_folders = [f for f in SOURCE_TEST_DIR.iterdir() if f.is_dir()]
    print(f"找到 {len(test_folders)} 个测试文件夹\n")
    
    # 处理每个文件夹
    results = []
    for folder in test_folders:
        output_folder = OUTPUT_TEST_DIR / folder.name
        result = process_single_folder(folder, output_folder)
        if result:
            results.append(result)
    
    # 打印汇总统计
    print("\n" + "="*60)
    print("处理汇总")
    print("="*60)
    
    total_folders = len(results)
    total_original_assoc = sum(r['original_associations'] for r in results)
    total_filtered_assoc = sum(r['filtered_associations'] for r in results)
    total_removed_assoc = sum(r['removed_associations'] for r in results)
    total_valve_elements = sum(r['total_valve_elements'] for r in results)
    total_unassoc_valves = sum(r['unassociated_valves'] for r in results)
    
    print(f"处理文件夹数: {total_folders}")
    print(f"原始关联总数: {total_original_assoc}")
    print(f"过滤后关联数: {total_filtered_assoc}")
    print(f"移除关联数: {total_removed_assoc} (仪表相关)")
    print(f"总阀门元件数: {total_valve_elements}")
    print(f"无关联阀门数: {total_unassoc_valves}")
    print()
    
    # 保存汇总结果
    summary_file = OUTPUT_TEST_DIR / "processing_summary.json"
    summary = {
        'total_folders': total_folders,
        'total_original_associations': total_original_assoc,
        'total_filtered_associations': total_filtered_assoc,
        'total_removed_associations': total_removed_assoc,
        'total_valve_elements': total_valve_elements,
        'total_unassociated_valves': total_unassoc_valves,
        'per_folder_results': results
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 处理完成！")
    print(f"📊 汇总信息已保存: {summary_file}")
    print(f"📁 统一后的测试集: {OUTPUT_TEST_DIR}")
    print()
    print("说明:")
    print("- 已去除所有仪表相关的关联关系")
    print("- 元件列表保持完整（包括无关联的阀门元件）")
    print("- KNN和Qwen现在可以使用相同的ground truth进行评估")
    print("="*60)


if __name__ == '__main__':
    main()

