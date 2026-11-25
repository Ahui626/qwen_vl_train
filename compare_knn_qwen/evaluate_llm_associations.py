"""
LLM关联关系评估脚本

功能：
1. 读取测试集中的标准元件框数据和关联关系
2. 对大图进行裁剪处理（类似训练数据预处理）
3. 使用LoRA微调后的模型进行推理
4. 计算准确率、召回率、F1等指标
5. 输出详细的评估报告和中间文件
"""

import os
import re
import json
import math
import torch
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from tqdm import tqdm
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import PeftModel


# ======================== 配置 ========================

# 设置GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

# 测试集目录
TEST_DATA_DIR = Path(__file__).parent / "测试集"

# LoRA模型路径（训练输出的最终模型）
LORA_MODEL_PATH = "/home/ubuntu/su/qwen_train/output/final"

# 基础模型路径
BASE_MODEL_PATH = "/home/ubuntu/cal_llm_test/test/models/Qwen2.5-VL-3B-Instruct"

# 中间文件保存目录
INTERMEDIATE_DIR = Path(__file__).parent / "llm_intermediate_files"

# 结果保存目录
RESULTS_DIR = Path(__file__).parent / "llm_results"

# 预处理配置（与训练时保持一致）
WINDOW_WIDTH = 1024
WINDOW_HEIGHT = 1024
OVERLAP_RATIO = 0.3
BOX_LINE_WIDTH = 3
FONT_SIZE = 18
DEVICE_PADDING = 100

# 有效类别（只处理可能产生关联关系的类别）
VALID_CATEGORIES = ['设备序号', '设备', '位号', '仪表', '符号', '属性标签']

# 关联关系类型配置
RELATION_TYPES = {
    '设备序号': ['设备'],
    '位号': ['设备', '仪表', '符号'],
    '符号': ['属性标签']
}

# 颜色配置
COLOR_MAIN = (255, 0, 0)      # 红色 - 主体元件
COLOR_CANDIDATE = (255, 255, 0)  # 黄色 - 候选元件
COLOR_ID = (0, 255, 0)        # 绿色 - ID标签

# =====================================================


def parse_image_coords(folder_path: Path) -> Optional[Tuple[float, float, float, float]]:
    """从图片文件名中解析坐标信息"""
    for file in folder_path.iterdir():
        if file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
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


def find_image_file(folder_path: Path) -> Optional[Path]:
    """查找文件夹中的图片文件"""
    for ext in ['.jpg', '.jpeg', '.png']:
        for file in folder_path.glob(f'*{ext}'):
            return file
    return None


def load_elements(txt_path: Path) -> List[Dict]:
    """加载元件标注框txt文件"""
    elements = []
    skipped_count = 0
    
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            parts = line.split(',')
            if len(parts) < 7:
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
                
                # 检查NaN/Inf
                if any(math.isnan(v) or math.isinf(v) for v in [x1, y1, x2, y2]):
                    skipped_count += 1
                    continue
                
                # 检查坐标合理性
                if abs(x2 - x1) < 0.001 or abs(y2 - y1) < 0.001:
                    skipped_count += 1
                    continue
                
                # 只保留有效类别
                if category not in VALID_CATEGORIES:
                    continue
                
                elements.append({
                    'id': elem_id,
                    'category': category,
                    'subclass': subclass,
                    'bbox': [x1, y1, x2, y2],
                })
                
            except (ValueError, IndexError):
                skipped_count += 1
                continue
    
    if skipped_count > 0:
        print(f"  跳过 {skipped_count} 个无效元件")
    
    return elements


def load_relations(txt_path: Path) -> List[Dict]:
    """
    加载关联关系txt文件
    
    格式：relation_id,relation_type,cat1,cat2,bbox1(4),bbox2(4),id1,id2
    """
    relations = []
    
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split(',')
            if len(parts) >= 14:
                try:
                    relations.append({
                        'relation_id': parts[0],
                        'relation_type': parts[1],
                        'element1_category': parts[2],
                        'element2_category': parts[3],
                        'element1_id': parts[12],
                        'element2_id': parts[13],
                    })
                except (ValueError, IndexError):
                    continue
    
    return relations


def world_to_image_coords(world_bbox: list, img_world_coords: tuple, img_size: tuple) -> Optional[list]:
    """世界坐标转像素坐标（Y轴翻转）"""
    if any(math.isnan(c) or math.isinf(c) for c in world_bbox):
        return None
    
    w_minX, w_minY, w_maxX, w_maxY = img_world_coords
    img_w, img_h = img_size
    
    if w_maxX == w_minX or w_maxY == w_minY:
        return None
    
    scale_x = img_w / (w_maxX - w_minX)
    scale_y = img_h / (w_maxY - w_minY)
    
    try:
        x1 = int((world_bbox[0] - w_minX) * scale_x)
        x2 = int((world_bbox[2] - w_minX) * scale_x)
        y1 = int(img_h - (world_bbox[3] - w_minY) * scale_y)
        y2 = int(img_h - (world_bbox[1] - w_minY) * scale_y)
        
        if x1 >= x2 or y1 >= y2:
            return None
        
        return [x1, y1, x2, y2]
    except (ValueError, OverflowError):
        return None




def draw_box_with_id(draw: ImageDraw.Draw, bbox: list, color: tuple, simple_id: int, font):
    """在图片上绘制标注框和ID"""
    x1, y1, x2, y2 = bbox
    
    # 绘制矩形框
    draw.rectangle([x1, y1, x2, y2], outline=color, width=BOX_LINE_WIDTH)
    
    # 绘制ID标签（绿色，左上角，带白色边框）
    id_text = str(simple_id)
    
    # 计算文本位置（框左上角）
    text_x = x1
    text_y = y1 - FONT_SIZE - 2  # 稍微向上偏移
    
    # 如果超出上边界，就放在框内
    if text_y < 0:
        text_y = y1 + 2
    
    # 绘制白色边框（描边效果）
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx != 0 or dy != 0:
                draw.text((text_x + dx, text_y + dy), id_text, fill=(255, 255, 255), font=font)
    
    # 绘制绿色ID
    draw.text((text_x, text_y), id_text, fill=COLOR_ID, font=font)


def build_prompt(main_element: dict, candidate_elements: list) -> str:
    """构建提示词（与训练时一致）"""
    candidate_ids = [str(e['simple_id']) for e in candidate_elements]
    candidates_str = ", ".join(candidate_ids)
    
    prompt = f"""你是工程图纸元件关联识别助手。
图中黄色框标注了不同元件，红色框是查询主体，框左上角绿色数字是元件ID。
黄色框ID: {candidates_str}
红色框与哪个黄色框有关联？（1-2个）多个用逗号分隔，只输出ID。"""
    
    return prompt


def is_bbox_inside_window(bbox: list, window: tuple, threshold: float = 1.0) -> bool:
    """检查元件框是否完全在窗口内（100%，与训练时一致）"""
    wx1, wy1, wx2, wy2 = window
    bx1, by1, bx2, by2 = bbox
    
    ix1 = max(wx1, bx1)
    iy1 = max(wy1, by1)
    ix2 = min(wx2, bx2)
    iy2 = min(wy2, by2)
    
    if ix2 <= ix1 or iy2 <= iy1:
        return False
    
    bbox_area = (bx2 - bx1) * (by2 - by1)
    if bbox_area <= 0:
        return False
    
    intersection_area = (ix2 - ix1) * (iy2 - iy1)
    return intersection_area / bbox_area >= threshold


def parse_model_output(output: str) -> list:
    """解析模型输出的ID列表"""
    output = output.strip()
    
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


def process_folder(
    folder_path: Path,
    model,
    processor,
    device: str
) -> Dict:
    """
    处理单个测试文件夹
    
    Returns:
        评估结果字典
    """
    folder_name = folder_path.name
    print(f"\n处理文件夹: {folder_name}")
    
    # 1. 查找文件
    image_file = find_image_file(folder_path)
    if not image_file:
        print("  ❌ 未找到图片文件")
        return None
    
    # 查找txt文件（不含_关联后缀）
    txt_files = [f for f in folder_path.glob('*.txt') if not f.name.endswith('_关联.txt')]
    if not txt_files:
        print("  ❌ 未找到元件标注文件")
        return None
    txt_file = txt_files[0]
    
    # 查找关联关系文件
    relation_file = folder_path / f"{txt_file.stem}_关联.txt"
    if not relation_file.exists():
        print("  ❌ 未找到关联关系文件")
        return None
    
    print(f"  图片: {image_file.name}")
    print(f"  元件标注: {txt_file.name}")
    print(f"  关联关系: {relation_file.name}")
    
    # 2. 解析图片坐标
    img_coords = parse_image_coords(folder_path)
    if not img_coords:
        print("  ❌ 无法从文件名解析图片坐标")
        return None
    
    # 3. 加载图片
    image = Image.open(image_file).convert('RGB')
    img_w, img_h = image.size
    print(f"  图片尺寸: {img_w} x {img_h}")
    
    # 4. 加载元件和关联关系
    elements = load_elements(txt_file)
    relations = load_relations(relation_file)
    print(f"  加载元件: {len(elements)} 个")
    print(f"  加载关联关系: {len(relations)} 条")
    
    if not elements or not relations:
        print("  ⚠️ 元件或关联关系为空，跳过")
        return None
    
    # 5. 世界坐标转像素坐标
    for elem in elements:
        pixel_bbox = world_to_image_coords(elem['bbox'], img_coords, (img_w, img_h))
        if pixel_bbox is None:
            elem['pixel_bbox'] = None
        else:
            elem['pixel_bbox'] = pixel_bbox
    
    # 过滤掉坐标转换失败的元件
    elements = [e for e in elements if e['pixel_bbox'] is not None]
    print(f"  有效元件: {len(elements)} 个")
    
    # 6. 按主体元件分组关联关系
    relations_by_main = defaultdict(list)
    for rel in relations:
        main_id = rel['element1_id']
        target_id = rel['element2_id']
        relations_by_main[main_id].append(target_id)
    
    # 7. 创建元件ID映射
    elem_by_id = {e['id']: e for e in elements}
    
    # 8. 滑动窗口裁剪 + 推理（与训练时逻辑一致）
    step_x = int(WINDOW_WIDTH * (1 - OVERLAP_RATIO))
    step_y = int(WINDOW_HEIGHT * (1 - OVERLAP_RATIO))
    
    all_predictions = []  # 所有推理结果
    processed_mains = set()  # 已处理的主体元件（避免重复）
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", FONT_SIZE)
    except:
        font = ImageFont.load_default()
    
    crop_count = 0
    
    # 滑动窗口遍历
    for y in range(0, max(1, img_h - WINDOW_HEIGHT + 1), step_y):
        for x in range(0, max(1, img_w - WINDOW_WIDTH + 1), step_x):
            actual_win_w = min(WINDOW_WIDTH, img_w - x)
            actual_win_h = min(WINDOW_HEIGHT, img_h - y)
            window = (x, y, x + actual_win_w, y + actual_win_h)
            
            # 对每个主体元件生成一个样本（包含其所有关联目标）
            for main_id, target_ids in relations_by_main.items():
                if main_id in processed_mains:
                    continue
                
                # 检查主体元件是否在窗口内
                if main_id not in elem_by_id:
                    continue
                main_elem = elem_by_id[main_id]
                
                if not is_bbox_inside_window(main_elem['pixel_bbox'], window):
                    continue
                
                # 检查所有关联目标是否在窗口内
                valid_target_ids = []
                for target_id in target_ids:
                    if target_id in elem_by_id:
                        target_elem = elem_by_id[target_id]
                        if is_bbox_inside_window(target_elem['pixel_bbox'], window):
                            valid_target_ids.append(target_id)
                
                if not valid_target_ids:
                    continue  # 没有有效的关联目标
                
                processed_mains.add(main_id)
                
                # 收集所有在窗口内的元件（除了主体元件）
                candidate_elements = []
                for elem in elements:
                    if elem['id'] == main_id:
                        continue  # 跳过主体元件
                    if is_bbox_inside_window(elem['pixel_bbox'], window):
                        candidate_elements.append(elem)
                
                if not candidate_elements:
                    continue  # 没有候选元件
                
                # 裁剪图片
                cropped_img = image.crop(window)
                draw = ImageDraw.Draw(cropped_img)
                
                # 分配简单ID
                # 候选元件：ID = 2, 3, 4...（黄色）
                target_simple_ids = []
                simple_id_map = {}
                
                for i, elem in enumerate(candidate_elements):
                    simple_id = i + 2  # 从2开始
                    elem['simple_id'] = simple_id
                    simple_id_map[elem['id']] = simple_id
                    
                    if elem['id'] in valid_target_ids:
                        target_simple_ids.append(simple_id)
                    
                    # 转换坐标到裁剪区域（相对坐标）
                    bbox = elem['pixel_bbox']
                    local_bbox = [int(bbox[0] - x), int(bbox[1] - y), int(bbox[2] - x), int(bbox[3] - y)]
                    
                    # 先画黄色框
                    draw.rectangle(local_bbox, outline=COLOR_CANDIDATE, width=BOX_LINE_WIDTH)
                
                # 主体元件：ID = 1（红色）
                main_bbox = main_elem['pixel_bbox']
                main_local_bbox = [int(main_bbox[0] - x), int(main_bbox[1] - y), int(main_bbox[2] - x), int(main_bbox[3] - y)]
                draw.rectangle(main_local_bbox, outline=COLOR_MAIN, width=BOX_LINE_WIDTH)
                
                # 画所有ID（确保ID盖在框上面）
                for elem in candidate_elements:
                    bbox = elem['pixel_bbox']
                    local_bbox = [int(bbox[0] - x), int(bbox[1] - y), int(bbox[2] - x), int(bbox[3] - y)]
                    draw_box_with_id(draw, local_bbox, COLOR_CANDIDATE, elem['simple_id'], font)
                
                draw_box_with_id(draw, main_local_bbox, COLOR_MAIN, 1, font)
                
                # 构建prompt（与训练时一致）
                main_element_info = {'simple_id': 1, 'category': main_elem['category']}
                candidate_elements_info = [
                    {'simple_id': e['simple_id'], 'category': e['category']} 
                    for e in candidate_elements
                ]
                prompt = build_prompt(main_element_info, candidate_elements_info)
                
                # 准备输入
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": prompt},
                        ],
                    },
                ]
                
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(
                    text=[text],
                    images=[cropped_img],
                    padding=True,
                    return_tensors="pt",
                ).to(device)
                
                # 推理
                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=128,
                        do_sample=False,
                    )
                
                # 解析输出
                generated_ids = output_ids[0][len(inputs.input_ids[0]):]
                output_text = processor.decode(generated_ids, skip_special_tokens=True)
                
                pred_simple_ids = parse_model_output(output_text)
                
                # 转换回原始ID
                pred_original_ids = []
                for sid in pred_simple_ids:
                    for orig_id, simple_id in simple_id_map.items():
                        if simple_id == sid:
                            pred_original_ids.append(orig_id)
                            break
                
                # 记录结果
                all_predictions.append({
                    'main_id': main_id,
                    'main_category': main_elem['category'],
                    'gt_target_ids': valid_target_ids,
                    'pred_target_ids': pred_original_ids,
                    'raw_output': output_text,
                    'crop_type': 'sliding',
                    'crop_region': window,
                })
                
                crop_count += 1
    
    # ==================== 设备单独处理 ====================
    # 对于未被滑动窗口完整捕获的设备元件，进行动态裁剪
    device_elements = [e for e in elements if e['category'] == '设备']
    
    for device_elem in device_elements:
        device_id = device_elem['id']
        
        # 跳过已处理的设备
        if device_id in processed_mains:
            continue
        
        # 检查该设备是否有关联关系
        if device_id not in relations_by_main:
            continue
        
        target_ids = relations_by_main[device_id]
        
        # 找出所有关联的目标元件
        target_elems = []
        for tid in target_ids:
            if tid in elem_by_id:
                target_elems.append(elem_by_id[tid])
        
        if not target_elems:
            continue
        
        # 计算包含设备和所有目标元件的最小边界框
        all_bboxes = [device_elem['pixel_bbox']] + [e['pixel_bbox'] for e in target_elems]
        
        min_x = min(bbox[0] for bbox in all_bboxes)
        min_y = min(bbox[1] for bbox in all_bboxes)
        max_x = max(bbox[2] for bbox in all_bboxes)
        max_y = max(bbox[3] for bbox in all_bboxes)
        
        # 添加padding
        min_x = max(0, min_x - DEVICE_PADDING)
        min_y = max(0, min_y - DEVICE_PADDING)
        max_x = min(img_w, max_x + DEVICE_PADDING)
        max_y = min(img_h, max_y + DEVICE_PADDING)
        
        # 创建动态窗口
        device_window = (int(min_x), int(min_y), int(max_x), int(max_y))
        
        # 检查窗口尺寸是否合理
        window_w = device_window[2] - device_window[0]
        window_h = device_window[3] - device_window[1]
        
        if window_w < 100 or window_h < 100:
            continue  # 窗口太小
        
        # 收集所有在设备窗口内的元件（除了设备本身）
        candidate_elements = []
        for elem in elements:
            if elem['id'] == device_id:
                continue
            if is_bbox_inside_window(elem['pixel_bbox'], device_window):
                candidate_elements.append(elem)
        
        if not candidate_elements:
            continue
        
        # 裁剪图片
        cropped_img = image.crop(device_window)
        draw = ImageDraw.Draw(cropped_img)
        
        # 分配简单ID
        target_simple_ids = []
        simple_id_map = {}
        
        for i, elem in enumerate(candidate_elements):
            simple_id = i + 2
            elem['simple_id'] = simple_id
            simple_id_map[elem['id']] = simple_id
            
            if elem['id'] in target_ids:
                target_simple_ids.append(simple_id)
            
            bbox = elem['pixel_bbox']
            local_bbox = [int(bbox[0] - min_x), int(bbox[1] - min_y), int(bbox[2] - min_x), int(bbox[3] - min_y)]
            draw.rectangle(local_bbox, outline=COLOR_CANDIDATE, width=BOX_LINE_WIDTH)
        
        # 设备元件用红色框
        device_bbox = device_elem['pixel_bbox']
        device_local_bbox = [int(device_bbox[0] - min_x), int(device_bbox[1] - min_y), 
                              int(device_bbox[2] - min_x), int(device_bbox[3] - min_y)]
        draw.rectangle(device_local_bbox, outline=COLOR_MAIN, width=BOX_LINE_WIDTH)
        
        # 画所有ID
        for elem in candidate_elements:
            bbox = elem['pixel_bbox']
            local_bbox = [int(bbox[0] - min_x), int(bbox[1] - min_y), int(bbox[2] - min_x), int(bbox[3] - min_y)]
            draw_box_with_id(draw, local_bbox, COLOR_CANDIDATE, elem['simple_id'], font)
        
        draw_box_with_id(draw, device_local_bbox, COLOR_MAIN, 1, font)
        
        # 构建prompt
        main_element_info = {'simple_id': 1, 'category': device_elem['category']}
        candidate_elements_info = [
            {'simple_id': e['simple_id'], 'category': e['category']} 
            for e in candidate_elements
        ]
        prompt = build_prompt(main_element_info, candidate_elements_info)
        
        # 准备输入
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text],
            images=[cropped_img],
            padding=True,
            return_tensors="pt",
        ).to(device)
        
        # 推理
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
            )
        
        # 解析输出
        generated_ids = output_ids[0][len(inputs.input_ids[0]):]
        output_text = processor.decode(generated_ids, skip_special_tokens=True)
        
        pred_simple_ids = parse_model_output(output_text)
        
        # 转换回原始ID
        pred_original_ids = []
        for sid in pred_simple_ids:
            for orig_id, simple_id in simple_id_map.items():
                if simple_id == sid:
                    pred_original_ids.append(orig_id)
                    break
        
        # 记录结果
        all_predictions.append({
            'main_id': device_id,
            'main_category': device_elem['category'],
            'gt_target_ids': target_ids,
            'pred_target_ids': pred_original_ids,
            'raw_output': output_text,
            'crop_type': 'device',
            'crop_region': device_window,
        })
        
        processed_mains.add(device_id)
        crop_count += 1
    
    print(f"  滑动窗口: {sum(1 for p in all_predictions if p['crop_type'] == 'sliding')} 个样本")
    print(f"  设备动态裁剪: {sum(1 for p in all_predictions if p['crop_type'] == 'device')} 个样本")
    print(f"  总预测数: {len(all_predictions)}")
    
    # 9. 计算指标
    total_tp = 0
    total_fp = 0
    total_fn = 0
    exact_match_count = 0
    
    for pred in all_predictions:
        pred_set = set(pred['pred_target_ids'])
        gt_set = set(pred['gt_target_ids'])
        
        tp = len(pred_set & gt_set)
        fp = len(pred_set - gt_set)
        fn = len(gt_set - pred_set)
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        if pred_set == gt_set:
            exact_match_count += 1
    
    # 计算整体指标
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    exact_match_rate = exact_match_count / len(all_predictions) if all_predictions else 0
    
    result = {
        'folder_name': folder_name,
        'total_samples': len(all_predictions),
        'true_positive': total_tp,
        'false_positive': total_fp,
        'false_negative': total_fn,
        'exact_match_count': exact_match_count,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'exact_match_rate': exact_match_rate,
        'predictions': all_predictions,
    }
    
    print(f"  ✅ TP={total_tp}, FP={total_fp}, FN={total_fn}")
    print(f"  📊 Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
    print(f"  🎯 Exact Match={exact_match_count}/{len(all_predictions)} ({exact_match_rate:.2%})")
    
    return result


def main():
    """主函数"""
    print("="*60)
    print("LLM关联关系评估")
    print("="*60)
    
    # 创建输出目录
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 查找所有测试文件夹
    test_folders = [f for f in TEST_DATA_DIR.iterdir() if f.is_dir()]
    test_folders.sort()
    
    print(f"\n找到 {len(test_folders)} 个测试文件夹")
    for folder in test_folders:
        print(f"  - {folder.name}")
    
    # 加载模型
    print(f"\n加载模型...")
    print(f"  基础模型: {BASE_MODEL_PATH}")
    print(f"  LoRA模型: {LORA_MODEL_PATH}")
    
    processor = AutoProcessor.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
    
    base_model = AutoModelForVision2Seq.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        load_in_8bit=True,
    )
    
    model = PeftModel.from_pretrained(base_model, LORA_MODEL_PATH)
    model.eval()
    
    device = next(model.parameters()).device
    print(f"  模型已加载到设备: {device}")
    
    # 处理每个文件夹
    all_results = []
    
    for folder in tqdm(test_folders, desc="评估进度"):
        result = process_folder(folder, model, processor, device)
        if result is not None:
            all_results.append(result)
            
            # 保存中间文件（详细预测结果）
            intermediate_file = INTERMEDIATE_DIR / f"{result['folder_name']}_predictions.json"
            with open(intermediate_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
    
    # 计算整体指标（微平均）
    print("\n" + "="*60)
    print("整体评估结果（微平均 - Micro Average）")
    print("="*60)
    
    total_tp = sum(r['true_positive'] for r in all_results)
    total_fp = sum(r['false_positive'] for r in all_results)
    total_fn = sum(r['false_negative'] for r in all_results)
    total_samples = sum(r['total_samples'] for r in all_results)
    total_exact_match = sum(r['exact_match_count'] for r in all_results)
    
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
    overall_exact_match_rate = total_exact_match / total_samples if total_samples > 0 else 0
    
    print(f"\n处理文件夹数: {len(all_results)}")
    print(f"总样本数: {total_samples}")
    print(f"True Positive: {total_tp}")
    print(f"False Positive: {total_fp}")
    print(f"False Negative: {total_fn}")
    print(f"Exact Match: {total_exact_match}/{total_samples} ({overall_exact_match_rate:.2%})")
    print(f"\nPrecision: {overall_precision:.4f}")
    print(f"Recall: {overall_recall:.4f}")
    print(f"F1 Score: {overall_f1:.4f}")
    
    # 保存汇总结果
    summary = {
        'total_folders': len(all_results),
        'total_samples': total_samples,
        'metrics': {
            'true_positive': total_tp,
            'false_positive': total_fp,
            'false_negative': total_fn,
            'exact_match_count': total_exact_match,
            'precision': overall_precision,
            'recall': overall_recall,
            'f1_score': overall_f1,
            'exact_match_rate': overall_exact_match_rate,
        },
        'per_folder_results': [
            {
                'folder_name': r['folder_name'],
                'total_samples': r['total_samples'],
                'true_positive': r['true_positive'],
                'false_positive': r['false_positive'],
                'false_negative': r['false_negative'],
                'exact_match_count': r['exact_match_count'],
                'precision': r['precision'],
                'recall': r['recall'],
                'f1': r['f1'],
                'exact_match_rate': r['exact_match_rate'],
            }
            for r in all_results
        ]
    }
    
    summary_file = RESULTS_DIR / "llm_evaluation_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 评估完成！")
    print(f"   汇总结果: {summary_file}")
    print(f"   中间文件: {INTERMEDIATE_DIR}/")
    print("="*60)


if __name__ == '__main__':
    main()

