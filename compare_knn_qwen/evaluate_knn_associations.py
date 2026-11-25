"""
KNN关联关系评估脚本

功能：
1. 读取测试集中的标准元件框数据（txt），转换成接口需要的JSON格式
2. 调用 calculate_associations 接口，获取KNN计算的关联关系
3. 读取标准关联关系（_关联.txt）
4. 计算准确率、召回率、F1等指标
"""

import os
import re
import json
import requests
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


# ======================== 配置 ========================

# 测试集目录（使用统一Ground Truth的版本）
# 目录结构：
#   测试集_统一GT/
#     ├── 10个项目_项目2_PSR15-73 四川天明_系统图_.../
#     │   ├── xxx.txt         (元件标注)
#     │   ├── xxx_关联.txt    (关联关系，已去除仪表)
#     │   ├── xxx.jpg         (图片，文件名含坐标)
#     │   └── xxx.json        (json数据)
#     ├── 10个项目_项目3_.../
#     └── ...
# 
# 注意：使用 prepare_unified_testset.py 生成统一后的测试集
# 如果目录不存在，请先运行: python prepare_unified_testset.py
TEST_DATA_DIR = Path(__file__).parent / "测试集_统一GT"

# 关联关系计算接口
API_URL = "http://192.168.3.226:5000/calculate_associations"

# 中间文件保存目录
INTERMEDIATE_DIR = Path(__file__).parent / "intermediate_files"

# 结果保存目录
RESULTS_DIR = Path(__file__).parent / "results"

# =====================================================


def parse_image_coords(folder_path: Path) -> Optional[Tuple[float, float, float, float]]:
    """
    从图片文件名中解析坐标信息
    文件名格式：xxx-(x_min,y_min,x_max,y_max).jpg
    
    Returns:
        (x_min, y_min, x_max, y_max) 或 None
    """
    for file in folder_path.iterdir():
        if file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            # 匹配坐标模式：-(...,...)
            match = re.search(r'-\(([^)]+)\)\.', file.name)
            if match:
                coords_str = match.group(1)
                parts = coords_str.split(',')
                if len(parts) == 4:
                    try:
                        return tuple(float(p) for p in parts)
                    except ValueError:
                        pass
    return None


def load_elements_txt(txt_path: Path) -> List[Dict]:
    """
    加载元件标注框txt文件
    
    格式：id,大类,小类,x1,y1,x2,y2
    
    OCR内容处理规则（用于关联逻辑）:
    - 属性标签: 生成 DN 规格（如 DN50），避免以 "Ø" 开头被过滤
      （根据utils.py：属性标签OCR内容以 "Ø" 开头会被排除，不参与符号-属性标签关联）
    - 位号/设备序号: 使用元件id前半部分作为content
    - 其他类别: content为空
    
    Returns:
        元件列表，每个元件包含 {id, category, subclass, x1, y1, x2, y2, content}
    """
    import math
    
    elements = []
    skipped_count = 0
    
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            parts = line.split(',')
            if len(parts) < 7:
                print(f"  警告: 第{line_num}行格式错误，跳过: {line[:50]}...")
                skipped_count += 1
                continue
            
            try:
                elem_id = parts[0]
                category = parts[1]
                subclass = parts[2]
                x1 = float(parts[3])
                y1 = float(parts[4])
                x2 = float(parts[5])
                y2 = float(parts[6])
                
                # 检查是否包含NaN或Inf值
                if any(math.isnan(v) or math.isinf(v) for v in [x1, y1, x2, y2]):
                    print(f"  警告: 第{line_num}行包含NaN或Inf值，跳过 (id={elem_id})")
                    skipped_count += 1
                    continue
                
                # 检查坐标合理性（bbox应该有面积）
                if abs(x2 - x1) < 0.001 or abs(y2 - y1) < 0.001:
                    print(f"  警告: 第{line_num}行坐标无效（宽或高为0），跳过 (id={elem_id})")
                    skipped_count += 1
                    continue
                
                # 构建接口需要的数据结构
                element = {
                    'id': elem_id,
                    'category': category,
                    'subclass': subclass,
                    'x1': x1,
                    'y1': y1,
                    'x2': x2,
                    'y2': y2,
                    'content': ''  # OCR内容，这里留空或简单赋值
                }
                
                # 对需要OCR的类别，简单赋值（用于关联逻辑判断）
                # 重要规则：属性标签如果OCR内容以 "Ø" 开头，会被排除不参与关联计算
                # 所以这里对属性标签生成合理的content，避免以Ø开头
                if category == '属性标签':
                    # 属性标签通常是管道规格，如 DN50, DN80, Φ57X3-CS 等
                    # 为了参与关联计算，我们生成不以Ø开头的内容
                    element['content'] = f"DN{50 + (hash(elem_id) % 50)}"  # 生成 DN50-DN100 之间的规格
                elif category in ['位号', '设备序号']:
                    # 简单赋值，使用id的一部分作为content
                    element['content'] = elem_id.split('-')[0] if '-' in elem_id else elem_id
                
                elements.append(element)
                
            except (ValueError, IndexError) as e:
                print(f"  警告: 第{line_num}行解析错误 ({e})，跳过: {line[:50]}...")
                skipped_count += 1
                continue
    
    if skipped_count > 0:
        print(f"  ⚠️  共跳过 {skipped_count} 行无效数据")
    
    return elements


def load_gt_associations(txt_path: Path) -> List[Dict]:
    """
    加载标准关联关系txt文件
    
    格式：关联id,关联类型,主体大类,关联元件大类,主体x1,主体y1,主体x2,主体y2,关联x1,关联y1,关联x2,关联y2,主体元件id,关联元件id
    
    Returns:
        关联关系列表
    """
    import math
    
    associations = []
    skipped_count = 0
    
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            parts = line.split(',')
            if len(parts) < 14:
                print(f"  警告: 关联文件第{line_num}行格式错误，跳过: {line[:50]}...")
                skipped_count += 1
                continue
            
            try:
                coords = [
                    float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7]),
                    float(parts[8]), float(parts[9]), float(parts[10]), float(parts[11])
                ]
                
                # 检查NaN/Inf值
                if any(math.isnan(v) or math.isinf(v) for v in coords):
                    print(f"  警告: 关联文件第{line_num}行包含NaN/Inf值，跳过 (id={parts[0]})")
                    skipped_count += 1
                    continue
                
                assoc = {
                    'assoc_id': parts[0],
                    'relation_type': parts[1],
                    'main_category': parts[2],
                    'target_category': parts[3],
                    'main_x1': coords[0],
                    'main_y1': coords[1],
                    'main_x2': coords[2],
                    'main_y2': coords[3],
                    'target_x1': coords[4],
                    'target_y1': coords[5],
                    'target_x2': coords[6],
                    'target_y2': coords[7],
                    'main_elem_id': parts[12],
                    'target_elem_id': parts[13]
                }
                associations.append(assoc)
                
            except (ValueError, IndexError) as e:
                print(f"  警告: 关联文件第{line_num}行解析错误 ({e})，跳过")
                skipped_count += 1
                continue
    
    if skipped_count > 0:
        print(f"  ⚠️  关联文件共跳过 {skipped_count} 行无效数据")
    
    return associations


def call_calculate_associations_api(
    classifications: List[Dict],
    x_min: float, y_min: float, x_max: float, y_max: float
) -> Optional[List[Dict]]:
    """
    调用关联关系计算接口
    
    Returns:
        关联关系列表 或 None（失败时）
    """
    import math
    
    try:
        # 再次检查数据中是否有NaN/Inf值（双重保险）
        for i, elem in enumerate(classifications):
            for key in ['x1', 'y1', 'x2', 'y2']:
                val = elem.get(key, 0)
                if math.isnan(val) or math.isinf(val):
                    print(f"  ⚠️  检测到元件 {i} ({elem.get('id')}) 的 {key} 值为 NaN/Inf，已修正为0")
                    elem[key] = 0.0
        
        # 检查坐标范围
        for coord_name, coord_val in [('x_min', x_min), ('y_min', y_min), 
                                       ('x_max', x_max), ('y_max', y_max)]:
            if math.isnan(coord_val) or math.isinf(coord_val):
                print(f"  ❌ 坐标参数 {coord_name} 值为 NaN/Inf: {coord_val}")
                return None
        
        payload = {
            'classifications': classifications,
            'x_min': x_min,
            'y_min': y_min,
            'x_max': x_max,
            'y_max': y_max
        }
        
        response = requests.post(
            API_URL,
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                return result.get('associations', [])
            else:
                print(f"  ❌ API返回错误: {result.get('error', 'unknown')}")
                return None
        else:
            print(f"  ❌ API请求失败: HTTP {response.status_code}")
            print(f"     响应内容: {response.text[:200]}")
            return None
            
    except requests.exceptions.ConnectionError:
        print(f"  ❌ 无法连接到API服务: {API_URL}")
        print(f"     请确保服务已启动: python app.py")
        return None
    except Exception as e:
        print(f"  ❌ API调用异常: {e}")
        import traceback
        traceback.print_exc()
        return None


def normalize_association(assoc: Dict) -> Tuple[str, str]:
    """
    归一化关联关系，用于比较
    
    接口返回的关联关系格式：
    {
        'source': 主体元件id,
        'target': 关联元件id,
        'type': 关联类型,
        'conf': 置信度
    }
    
    Returns:
        (主体元件id, 关联元件id)
    """
    # 接口使用 source 和 target 字段
    main_id = assoc.get('source', '')
    target_id = assoc.get('target', '')
    
    return (main_id, target_id)


def normalize_gt_association(assoc: Dict) -> Tuple[str, str]:
    """
    归一化标准关联关系
    
    Returns:
        (主体元件id, 关联元件id)
    """
    return (assoc['main_elem_id'], assoc['target_elem_id'])


def calculate_metrics(
    pred_associations: List[Dict],
    gt_associations: List[Dict]
) -> Dict:
    """
    计算评估指标
    
    Args:
        pred_associations: 接口预测的关联关系
        gt_associations: 标准关联关系
        
    Returns:
        包含各项指标的字典
    """
    # 归一化关联关系为集合，便于比较
    pred_set = set()
    for assoc in pred_associations:
        pair = normalize_association(assoc)
        if pair[0] and pair[1]:  # 确保id有效
            pred_set.add(pair)
            # 也添加反向关系（如果关联关系是无向的）
            # pred_set.add((pair[1], pair[0]))
    
    gt_set = set()
    for assoc in gt_associations:
        pair = normalize_gt_association(assoc)
        gt_set.add(pair)
    
    # 计算TP, FP, FN
    tp = len(pred_set & gt_set)  # 正确预测
    fp = len(pred_set - gt_set)  # 误报（预测了但不应该）
    fn = len(gt_set - pred_set)  # 漏报（应该预测但没有）
    
    # 计算指标
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'total_pred': len(pred_set),
        'total_gt': len(gt_set),
        'true_positive': tp,
        'false_positive': fp,
        'false_negative': fn,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'pred_set': pred_set,
        'gt_set': gt_set
    }


def process_single_folder(folder_path: Path, save_intermediate: bool = True) -> Optional[Dict]:
    """
    处理单个测试文件夹
    
    Args:
        folder_path: 测试数据文件夹路径
        save_intermediate: 是否保存中间文件
        
    Returns:
        评估结果字典 或 None（失败时）
    """
    folder_name = folder_path.name
    print(f"\n{'='*60}")
    print(f"处理: {folder_name}")
    print('='*60)
    
    # 1. 解析图片坐标
    coords = parse_image_coords(folder_path)
    if not coords:
        print(f"❌ 无法从图片文件名解析坐标")
        return None
    
    x_min, y_min, x_max, y_max = coords
    print(f"坐标范围: x=[{x_min}, {x_max}], y=[{y_min}, {y_max}]")
    
    # 2. 查找元件标注文件
    elements_txt = folder_path / f"{folder_name}.txt"
    if not elements_txt.exists():
        print(f"❌ 元件标注文件不存在: {elements_txt.name}")
        return None
    
    # 3. 查找标准关联关系文件
    gt_assoc_txt = folder_path / f"{folder_name}_关联.txt"
    if not gt_assoc_txt.exists():
        print(f"❌ 关联关系文件不存在: {gt_assoc_txt.name}")
        return None
    
    # 4. 加载元件数据
    print(f"加载元件标注...")
    elements = load_elements_txt(elements_txt)
    print(f"✅ 有效元件数量: {len(elements)}")
    
    if not elements:
        print(f"❌ 没有加载到任何有效元件")
        return None
    
    # 5. 保存中间文件（接口输入格式的JSON）
    if save_intermediate:
        INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
        
        intermediate_file = INTERMEDIATE_DIR / f"{folder_name}_classifications.json"
        intermediate_data = {
            'folder_name': folder_name,
            'coords': {
                'x_min': x_min,
                'y_min': y_min,
                'x_max': x_max,
                'y_max': y_max
            },
            'classifications': elements
        }
        
        with open(intermediate_file, 'w', encoding='utf-8') as f:
            json.dump(intermediate_data, f, ensure_ascii=False, indent=2)
        print(f"✅ 中间文件已保存: {intermediate_file.name}")
    
    # 6. 调用API计算关联关系
    print(f"\n调用关联关系计算接口...")
    pred_associations = call_calculate_associations_api(
        elements, x_min, y_min, x_max, y_max
    )
    
    if pred_associations is None:
        print(f"❌ API调用失败，跳过此文件夹")
        return None
    
    print(f"✅ API返回关联数量: {len(pred_associations)}")
    
    # 7. 保存API返回结果
    if save_intermediate:
        api_result_file = INTERMEDIATE_DIR / f"{folder_name}_api_result.json"
        with open(api_result_file, 'w', encoding='utf-8') as f:
            json.dump(pred_associations, f, ensure_ascii=False, indent=2)
        print(f"✅ API结果已保存: {api_result_file.name}")
    
    # 8. 加载标准关联关系
    print(f"\n加载标准关联关系...")
    gt_associations = load_gt_associations(gt_assoc_txt)
    print(f"✅ 标准关联数量: {len(gt_associations)}")
    
    # 9. 计算评估指标
    metrics = calculate_metrics(pred_associations, gt_associations)
    
    print(f"\n📊 评估结果:")
    print(f"  预测关联数: {metrics['total_pred']}")
    print(f"  标准关联数: {metrics['total_gt']}")
    print(f"  正确预测(TP): {metrics['true_positive']}")
    print(f"  误报(FP): {metrics['false_positive']}")
    print(f"  漏报(FN): {metrics['false_negative']}")
    print(f"  准确率(Precision): {metrics['precision']:.4f}")
    print(f"  召回率(Recall): {metrics['recall']:.4f}")
    print(f"  F1分数: {metrics['f1_score']:.4f}")
    
    return {
        'folder_name': folder_name,
        'coords': coords,
        'num_elements': len(elements),
        'num_pred_assoc': len(pred_associations),
        'num_gt_assoc': len(gt_associations),
        'metrics': {
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score'],
            'tp': metrics['true_positive'],
            'fp': metrics['false_positive'],
            'fn': metrics['false_negative']
        }
    }


def evaluate_all(test_dir: Path = None, save_intermediate: bool = True) -> Dict:
    """
    评估所有测试文件夹
    
    Args:
        test_dir: 测试数据目录，默认使用 TEST_DATA_DIR
        save_intermediate: 是否保存中间文件
        
    Returns:
        汇总评估结果
    """
    if test_dir is None:
        test_dir = TEST_DATA_DIR
    
    print("="*60)
    print("KNN关联关系评估")
    print("="*60)
    print(f"测试目录: {test_dir}")
    print(f"API地址: {API_URL}")
    print()
    
    # 创建结果目录
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    
    # 收集所有测试文件夹
    test_folders = [f for f in test_dir.iterdir() if f.is_dir()]
    print(f"找到 {len(test_folders)} 个测试文件夹:")
    for i, folder in enumerate(test_folders, 1):
        print(f"  {i}. {folder.name}")
    print()
    
    # 处理每个文件夹
    results = []
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for folder in test_folders:
        result = process_single_folder(folder, save_intermediate)
        if result:
            results.append(result)
            total_tp += result['metrics']['tp']
            total_fp += result['metrics']['fp']
            total_fn += result['metrics']['fn']
    
    # 计算总体指标
    total_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    total_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    total_f1 = 2 * total_precision * total_recall / (total_precision + total_recall) if (total_precision + total_recall) > 0 else 0.0
    
    # 汇总结果
    summary = {
        'num_folders': len(results),
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_fn': total_fn,
        'micro_precision': total_precision,
        'micro_recall': total_recall,
        'micro_f1': total_f1,
        'folder_results': results
    }
    
    # 保存汇总结果
    summary_file = RESULTS_DIR / "knn_evaluation_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    # 打印汇总
    print("\n" + "="*60)
    print("📊 总体评估结果")
    print("="*60)
    print(f"处理文件夹数: {len(results)}")
    print(f"总正确预测(TP): {total_tp}")
    print(f"总误报(FP): {total_fp}")
    print(f"总漏报(FN): {total_fn}")
    print(f"微平均准确率: {total_precision:.4f}")
    print(f"微平均召回率: {total_recall:.4f}")
    print(f"微平均F1分数: {total_f1:.4f}")
    print(f"\n结果已保存: {summary_file}")
    
    return summary


def evaluate_single(folder_name: str, save_intermediate: bool = True) -> Optional[Dict]:
    """
    评估单个测试文件夹（便捷函数）
    
    Args:
        folder_name: 文件夹名称
        save_intermediate: 是否保存中间文件
        
    Returns:
        评估结果
    """
    folder_path = TEST_DATA_DIR / folder_name
    if not folder_path.exists():
        print(f"❌ 文件夹不存在: {folder_path}")
        return None
    
    return process_single_folder(folder_path, save_intermediate)


if __name__ == '__main__':
    """
    KNN关联关系评估脚本（使用与Qwen统一的Ground Truth）
    
    使用方法:
    
    1. 首次运行前，先生成统一测试集:
       python prepare_unified_testset.py
    
    2. 评估所有测试文件夹:
       python evaluate_knn_associations.py
    
    3. 在代码中评估单个文件夹:
       result = evaluate_single("10个项目_项目2_PSR15-73 四川天明_系统图_天明电厂03--精处理系统系统 - 运行")
    
    数据说明:
    - 使用 测试集_统一GT/ 目录（由 prepare_unified_testset.py 生成）
    - 已去除所有仪表相关的关联关系
    - 元件列表保持完整（包括无关联的阀门元件）
    - 与Qwen评估使用相同的ground truth标准
    
    目录结构:
    - 测试集_统一GT/              (TEST_DATA_DIR)
      - 子文件夹1/                (包含 txt, jpg, json, _关联.txt)
      - 子文件夹2/
      - ...
    
    OCR内容处理:
    - 属性标签: 自动生成 DN 规格（避免以 Ø 开头被过滤）
    - 位号/设备序号: 使用元件id生成
    
    注意:
    - 运行前请确保 KNN app.py 服务已启动 (http://127.0.0.1:5000)
    - 中间文件会保存在 intermediate_files/ 目录
    - 结果会保存在 results/ 目录
    """
    
    # 评估所有测试文件夹
    summary = evaluate_all()

