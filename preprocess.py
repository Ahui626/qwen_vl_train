"""
数据预处理脚本 - 位号中心裁剪版本
以位号元件为中心，固定1024尺寸裁剪
反转关联方向：位号 → 符号/设备/属性标签等

用法: python preprocess.py --config config.yaml
"""
import os
import re
import json
import yaml
import argparse
import math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from collections import defaultdict


def load_config(config_path: str) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def parse_image_coords(image_path: str) -> tuple:
    """从图片文件名解析坐标"""
    match = re.search(r'\((-?[\d.]+),(-?[\d.]+),(-?[\d.]+),(-?[\d.]+)\)', str(image_path))
    if match:
        return tuple(map(float, match.groups()))
    return None


def load_elements(elements_file: str, valid_categories: list = None) -> list:
    """加载元件标注"""
    elements = []
    with open(elements_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) >= 7:
                try:
                    category = parts[1]
                    if valid_categories and category not in valid_categories:
                        continue
                    bbox = [float(parts[i]) for i in range(3, 7)]
                    if any(math.isnan(c) or math.isinf(c) for c in bbox):
                        continue
                    if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
                        continue
                    elements.append({
                        'id': parts[0],
                        'category': category,
                        'sub_category': parts[2],
                        'bbox': bbox,
                    })
                except (ValueError, IndexError):
                    continue
    return elements


def load_relations(relation_file: str) -> list:
    """加载关联关系"""
    relations = []
    with open(relation_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) >= 14:
                relations.append({
                    'relation_type': parts[1],
                    'element1_category': parts[2],
                    'element2_category': parts[3],
                    'element1_id': parts[12],
                    'element2_id': parts[13],
                })
    return relations


def world_to_image_coords(world_bbox: list, img_world_coords: tuple, img_size: tuple) -> list:
    """世界坐标转像素坐标"""
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
        return [x1, y1, x2, y2]
    except:
        return None


def get_font(size: int = 16):
    """获取字体"""
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, size)
            except:
                continue
    return ImageFont.load_default()


def is_bbox_inside_window(bbox: list, window: tuple, margin: float = 1.0) -> bool:
    """检查元件框是否完全在窗口内（100%）"""
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
    return intersection_area / bbox_area >= margin


def draw_box(draw, bbox, color, line_width):
    """仅绘制边框"""
    draw.rectangle(bbox, outline=color, width=line_width)


def draw_id_label(draw, bbox, simple_id, font):
    """
    绘制ID标注
    ID统一放在框的左上角，数字盖在框上面
    """
    text = str(simple_id)
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]
    
    box_x1, box_y1, box_x2, box_y2 = bbox
    
    text_x = box_x1 - text_w // 2
    text_y = box_y1 - text_h // 2
    
    if text_x < 0:
        text_x = box_x1 + 2
    if text_y < 0:
        text_y = box_y1 + 2
    
    outline_color = (255, 255, 255)
    outline_width = 2
    for dx in range(-outline_width, outline_width + 1):
        for dy in range(-outline_width, outline_width + 1):
            if dx != 0 or dy != 0:
                draw.text((text_x + dx, text_y + dy), text, fill=outline_color, font=font)
    
    draw.text((text_x, text_y), text, fill=(0, 255, 0), font=font)


def calculate_bbox_distance(bbox1: list, bbox2: list) -> tuple:
    """
    计算两个bbox之间的边界距离（模拟从bbox1向外扩展碰到bbox2的距离）
    返回: (distance, is_inside)
    - is_inside: bbox2是否在bbox1内部（优先级最高，距离为负数）
    - distance: 边界之间的最小距离
      - 如果bbox2在bbox1内部：负数（越深入负得越多）
      - 如果两个框重叠：0
      - 如果两个框分离：最近边界的距离
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # 检查bbox2是否完全在bbox1内部
    is_inside = (x1_2 >= x1_1 and y1_2 >= y1_1 and x2_2 <= x2_1 and y2_2 <= y2_1)
    
    if is_inside:
        # 计算内部深度（到边界的最小距离，越深入负得越多）
        depth_left = x1_2 - x1_1
        depth_right = x2_1 - x2_2
        depth_top = y1_2 - y1_1
        depth_bottom = y2_1 - y2_2
        min_depth = min(depth_left, depth_right, depth_top, depth_bottom)
        distance = -min_depth  # 负数表示内部
        return distance, True
    
    # 计算边界距离（矩形到矩形的最小距离）
    # 水平方向距离
    if x2_1 < x1_2:  # bbox1在bbox2左侧
        dx = x1_2 - x2_1
    elif x2_2 < x1_1:  # bbox1在bbox2右侧
        dx = x1_1 - x2_2
    else:  # 水平方向有重叠
        dx = 0
    
    # 垂直方向距离
    if y2_1 < y1_2:  # bbox1在bbox2上方
        dy = y1_2 - y2_1
    elif y2_2 < y1_1:  # bbox1在bbox2下方
        dy = y1_1 - y2_2
    else:  # 垂直方向有重叠
        dy = 0
    
    # 总距离
    if dx == 0 and dy == 0:
        # 两个框重叠但bbox2不在bbox1内部
        distance = 0
    elif dx == 0:
        # 水平对齐，只有垂直距离
        distance = dy
    elif dy == 0:
        # 垂直对齐，只有水平距离
        distance = dx
    else:
        # 对角距离（边界到边界的欧氏距离）
        distance = math.sqrt(dx**2 + dy**2)
    
    return distance, False


def process_main_element(
    main_elem_id: str,
    target_elem_ids: list,
    elements: list,
    element_map: dict,
    image: Image,
    window: tuple,
    line_width: int,
    font,
    max_candidates: int = 7,
) -> dict:
    """
    处理一个主体元件及其所有关联目标，生成训练数据
    
    设计：
    - 红色框(ID=1)：主体元件
    - 黄色框(ID=2,3,4...)：候选元件（最多7个，基于边界距离筛选）
    - 任务：找出红色框与哪些黄色框有关联（1-2个）
    
    距离计算：模拟从主体框向外扩展，按碰到候选框的顺序排序
    """
    x, y, x2, y2 = window
    
    if main_elem_id not in element_map:
        return None
    main_elem = element_map[main_elem_id]
    
    if not is_bbox_inside_window(main_elem['pixel_bbox'], window):
        return None
    
    valid_target_ids = []
    for target_id in target_elem_ids:
        if target_id in element_map:
            target_elem = element_map[target_id]
            if is_bbox_inside_window(target_elem['pixel_bbox'], window):
                valid_target_ids.append(target_id)
    
    # 允许valid_target_ids为空（没有关联关系的情况）
    
    cropped = image.crop(window)
    draw = ImageDraw.Draw(cropped)
    
    # 收集所有在窗口内的元件（除了主体元件），并计算距离
    all_candidates = []
    main_bbox = main_elem['pixel_bbox']
    
    for elem in elements:
        if elem['id'] == main_elem_id:
            continue
        if is_bbox_inside_window(elem['pixel_bbox'], window):
            # 计算与主体元件的距离
            distance, is_inside = calculate_bbox_distance(main_bbox, elem['pixel_bbox'])
            
            px1, py1, px2, py2 = elem['pixel_bbox']
            relative_bbox = [int(px1 - x), int(py1 - y), int(px2 - x), int(py2 - y)]
            
            all_candidates.append({
                'original_id': elem['id'],
                'category': elem['category'],
                'bbox': relative_bbox,
                'is_target': elem['id'] in valid_target_ids,
                'distance': distance,
                'is_inside': is_inside,
            })
    
    # 按距离排序筛选候选元件：
    # 1. 优先选择在主体内部的元件（is_inside=True）
    # 2. 然后按距离从近到远排序
    # 3. 确保所有目标元件都被包含
    # 4. 最多选择max_candidates个
    
    # 先确保所有目标元件都在候选列表中
    target_candidates = [c for c in all_candidates if c['is_target']]
    non_target_candidates = [c for c in all_candidates if not c['is_target']]
    
    # 对非目标候选元件排序：内部元件优先，然后按距离排序
    non_target_candidates.sort(key=lambda c: (not c['is_inside'], c['distance']))
    
    # 选择候选元件：目标元件 + 最近的非目标元件
    remaining_slots = max_candidates - len(target_candidates)
    if remaining_slots > 0:
        candidate_elements = target_candidates + non_target_candidates[:remaining_slots]
    else:
        # 如果目标元件已经超过max_candidates，全部保留
        candidate_elements = target_candidates
    
    # 如果没有任何候选元件，也可以生成样本（用于告诉模型可能没有关联）
    # 只要有主体元件即可
    
    # 重新按距离排序（用于分配ID）
    candidate_elements.sort(key=lambda c: (not c['is_inside'], c['distance']))
    
    # 主体元件的相对坐标
    px1, py1, px2, py2 = main_elem['pixel_bbox']
    main_bbox = [int(px1 - x), int(py1 - y), int(px2 - x), int(py2 - y)]
    
    target_simple_ids = []
    for i, elem in enumerate(candidate_elements):
        elem['simple_id'] = i + 2
        if elem['is_target']:
            target_simple_ids.append(elem['simple_id'])
    
    # 允许target_simple_ids为空（没有关联关系的情况）
    
    RED = (255, 0, 0)
    YELLOW = (255, 255, 0)
    
    for elem in candidate_elements:
        draw_box(draw, elem['bbox'], YELLOW, line_width)
    draw_box(draw, main_bbox, RED, line_width)
    
    for elem in candidate_elements:
        draw_id_label(draw, elem['bbox'], elem['simple_id'], font)
    draw_id_label(draw, main_bbox, 1, font)
    
    main_element_info = {
        'simple_id': 1,
        'category': main_elem['category'],
        'bbox': main_bbox,
    }
    
    candidate_elements_info = []
    for elem in candidate_elements:
        candidate_elements_info.append({
            'simple_id': elem['simple_id'],
            'category': elem['category'],
            'bbox': elem['bbox'],
        })
    
    return {
        'cropped_image': cropped,
        'main_element': main_element_info,
        'candidate_elements': sorted(candidate_elements_info, key=lambda x: x['simple_id']),
        'answer': {
            'target_ids': sorted(target_simple_ids),
        },
    }


def is_valve_element(elem: dict) -> bool:
    """判断是否为阀门类元件（符号类且小类别名包含"阀"字）"""
    return elem['category'] == '符号' and '阀' in elem['sub_category']


def create_centered_window(elem_bbox: list, crop_size: int, img_size: tuple) -> tuple:
    """
    创建以元件为中心的固定尺寸窗口
    
    Args:
        elem_bbox: 元件的像素坐标 [x1, y1, x2, y2]
        crop_size: 裁剪尺寸（正方形）
        img_size: 图片尺寸 (width, height)
    
    Returns:
        窗口坐标 (x1, y1, x2, y2)
    """
    img_w, img_h = img_size
    
    # 计算元件中心
    elem_cx = (elem_bbox[0] + elem_bbox[2]) / 2
    elem_cy = (elem_bbox[1] + elem_bbox[3]) / 2
    
    # 计算窗口左上角
    win_x1 = elem_cx - crop_size / 2
    win_y1 = elem_cy - crop_size / 2
    
    # 边界检查和调整
    if win_x1 < 0:
        win_x1 = 0
    if win_y1 < 0:
        win_y1 = 0
    if win_x1 + crop_size > img_w:
        win_x1 = img_w - crop_size
    if win_y1 + crop_size > img_h:
        win_y1 = img_h - crop_size
    
    # 再次确保不越界
    win_x1 = max(0, win_x1)
    win_y1 = max(0, win_y1)
    
    # 计算实际窗口尺寸
    actual_w = min(crop_size, img_w - win_x1)
    actual_h = min(crop_size, img_h - win_y1)
    
    return (int(win_x1), int(win_y1), int(win_x1 + actual_w), int(win_y1 + actual_h))


def process_folder(folder: Path, config: dict, output_base: Path) -> list:
    """处理单个数据文件夹"""
    preprocess_cfg = config['preprocess']
    line_width = preprocess_cfg.get('box_line_width', 3)
    font_size = preprocess_cfg.get('font_size', 20)
    font = get_font(font_size)
    crop_size = preprocess_cfg.get('valve_crop_size', 1024)
    
    jpg_files = list(folder.glob('*.jpg'))
    if not jpg_files:
        return []
    image_path = jpg_files[0]
    
    img_world_coords = parse_image_coords(str(image_path))
    if not img_world_coords:
        return []
    
    txt_files = [f for f in folder.glob('*.txt') if not f.name.endswith('_关联.txt')]
    if not txt_files:
        return []
    
    valid_categories = preprocess_cfg.get('valid_categories', None)
    elements = load_elements(str(txt_files[0]), valid_categories)
    if not elements:
        return []
    
    relation_files = list(folder.glob('*_关联.txt'))
    if not relation_files:
        return []
    relations = load_relations(str(relation_files[0]))
    if not relations:
        return []
    
    try:
        image = Image.open(image_path).convert('RGB')
        img_w, img_h = image.size
    except:
        return []
    
    for elem in elements:
        pixel_bbox = world_to_image_coords(elem['bbox'], img_world_coords, image.size)
        if pixel_bbox:
            elem['pixel_bbox'] = pixel_bbox
    
    elements = [e for e in elements if 'pixel_bbox' in e]
    element_map = {e['id']: e for e in elements}
    
    relations = [r for r in relations if r['element1_id'] in element_map and r['element2_id'] in element_map]
    
    # 过滤掉包含仪表的关联关系
    relations = [r for r in relations if r['element1_category'] != '仪表' and r['element2_category'] != '仪表']
    
    # 步骤1：构建完整的关联图（element -> 所有直接关联的elements）
    element_graph = defaultdict(set)
    for rel in relations:
        elem1_id = rel['element1_id']
        elem2_id = rel['element2_id']
        # 双向建图
        element_graph[elem1_id].add(elem2_id)
        element_graph[elem2_id].add(elem1_id)
    
    # 步骤2：对于每个位号，收集其所有应该关联的元件
    # 包括：直接关联的符号/设备，以及这些符号/设备关联的其他元件（属性标签等）
    main_to_targets = defaultdict(list)
    
    # 遍历所有位号元件
    for elem in elements:
        if elem['category'] != '位号':
            continue
        
        tag_id = elem['id']
        target_ids = set()
        
        # 找到位号直接关联的所有元件
        direct_neighbors = element_graph.get(tag_id, set())
        
        for neighbor_id in direct_neighbors:
            neighbor_elem = element_map.get(neighbor_id)
            if not neighbor_elem:
                continue
            
            neighbor_cat = neighbor_elem['category']
            
            # 如果是符号或设备，添加它，并收集它关联的所有元件
            if neighbor_cat in ['符号', '设备']:
                target_ids.add(neighbor_id)
                
                # 收集该符号/设备关联的所有其他元件（属性标签、设备序号等）
                secondary_neighbors = element_graph.get(neighbor_id, set())
                for secondary_id in secondary_neighbors:
                    if secondary_id == tag_id:  # 跳过位号自己
                        continue
                    secondary_elem = element_map.get(secondary_id)
                    if secondary_elem and secondary_elem['category'] not in ['位号']:  # 不包括其他位号
                        target_ids.add(secondary_id)
            
            # 如果直接关联的是属性标签、设备序号等，也添加
            elif neighbor_cat in ['属性标签', '设备序号']:
                target_ids.add(neighbor_id)
        
        # 保存该位号的所有关联目标
        if target_ids:
            main_to_targets[tag_id] = list(target_ids)
    
    folder_output = output_base / folder.name
    folder_output.mkdir(parents=True, exist_ok=True)
    
    samples = []
    device_sample_idx = 0  # 设备样本计数器
    tag_sample_idx = 0     # 位号样本计数器
    processed_tag_ids = set()  # 已处理的位号ID
    
    # ==================== 设备单独处理（优先）====================
    # 先处理设备，并标记设备内的位号为已使用
    device_padding = preprocess_cfg.get('device_padding', 100)
    device_elements = [e for e in elements if e['category'] == '设备']
    
    # 反向映射：设备 -> 关联的位号
    device_to_tags = defaultdict(list)
    for tag_id, target_ids in main_to_targets.items():
        tag_elem = element_map.get(tag_id)
        if not tag_elem or tag_elem['category'] != '位号':
            continue
        for target_id in target_ids:
            target_elem = element_map.get(target_id)
            if target_elem and target_elem['category'] == '设备':
                device_to_tags[target_id].append(tag_id)
    
    for device_elem in device_elements:
        device_id = device_elem['id']
        
        # 获取该设备关联的位号
        related_tag_ids = device_to_tags.get(device_id, [])
        if not related_tag_ids:
            # 设备里没有位号，跳过
            continue
        
        # 收集设备和所有相关位号的目标元件
        all_related_elem_ids = set([device_id])
        for tag_id in related_tag_ids:
            all_related_elem_ids.update(main_to_targets.get(tag_id, []))
        
        # 计算所有相关元件的bbox范围
        related_elems = [element_map[eid] for eid in all_related_elem_ids if eid in element_map]
        if not related_elems:
            continue
        
        all_bboxes = [e['pixel_bbox'] for e in related_elems]
        min_x = min(bbox[0] for bbox in all_bboxes)
        min_y = min(bbox[1] for bbox in all_bboxes)
        max_x = max(bbox[2] for bbox in all_bboxes)
        max_y = max(bbox[3] for bbox in all_bboxes)
        
        # 添加padding
        min_x = max(0, min_x - device_padding)
        min_y = max(0, min_y - device_padding)
        max_x = min(img_w, max_x + device_padding)
        max_y = min(img_h, max_y + device_padding)
        
        device_window = (int(min_x), int(min_y), int(max_x), int(max_y))
        window_w = device_window[2] - device_window[0]
        window_h = device_window[3] - device_window[1]
        
        if window_w < 100 or window_h < 100:
            continue
        
        # 调整窗口尺寸：如果小于1024则扩展到1024，大于1024则保持实际尺寸
        min_size = 1024
        if window_w < min_size or window_h < min_size:
            # 计算需要扩展的量
            expand_w = max(0, min_size - window_w)
            expand_h = max(0, min_size - window_h)
            
            # 在中心扩展
            expand_left = expand_w // 2
            expand_right = expand_w - expand_left
            expand_top = expand_h // 2
            expand_bottom = expand_h - expand_top
            
            # 应用扩展，确保不超出图像边界
            new_min_x = max(0, min_x - expand_left)
            new_min_y = max(0, min_y - expand_top)
            new_max_x = min(img_w, max_x + expand_right)
            new_max_y = min(img_h, max_y + expand_bottom)
            
            # 如果一边触及边界导致尺寸不足，从另一边补足
            actual_w = new_max_x - new_min_x
            actual_h = new_max_y - new_min_y
            
            if actual_w < min_size and new_max_x < img_w:
                new_max_x = min(img_w, new_min_x + min_size)
            if actual_w < min_size and new_min_x > 0:
                new_min_x = max(0, new_max_x - min_size)
                
            if actual_h < min_size and new_max_y < img_h:
                new_max_y = min(img_h, new_min_y + min_size)
            if actual_h < min_size and new_min_y > 0:
                new_min_y = max(0, new_max_y - min_size)
            
            device_window = (int(new_min_x), int(new_min_y), int(new_max_x), int(new_max_y))
        
        # 对每个关联的位号生成样本（主体依然是位号）
        for tag_id in related_tag_ids:
            tag_elem = element_map.get(tag_id)
            if not tag_elem:
                continue
            
            # 标记位号为已处理
            processed_tag_ids.add(tag_id)
            
            # 获取该位号的关联目标
            target_ids = main_to_targets.get(tag_id, [])
            
            # 使用设备动态窗口处理
            result = process_main_element(
                tag_id, target_ids, elements, element_map, image, device_window, line_width, font
            )
            
            if result is None:
                continue
            
            # 保存图片（使用device_前缀）
            img_filename = f"device_{device_sample_idx:04d}.jpg"
            img_path = folder_output / img_filename
            result['cropped_image'].save(img_path, quality=95)
            
            # 保存JSON
            json_filename = f"device_{device_sample_idx:04d}.json"
            json_path = folder_output / json_filename
            sample_data = {
                'image': img_filename,
                'source_folder': folder.name,
                'crop_type': 'device_expanded',  # 设备扩展裁剪
                'main_element': result['main_element'],
                'candidate_elements': result['candidate_elements'],
                'answer': result['answer'],
            }
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(sample_data, f, ensure_ascii=False, indent=2)
            
            num_targets = len(result['answer']['target_ids'])
            sample_info = {
                'json_path': str(json_path),
                'image_path': str(img_path),
                'num_targets': num_targets,
                'crop_type': 'device_expanded',
            }
            samples.append(sample_info)
            
            device_sample_idx += 1
    
    # ==================== 位号元件处理 ====================
    # 遍历剩余的位号元件（跳过已在设备中处理的）
    tag_elements = [e for e in elements if e['category'] == '位号']
    
    for tag_elem in tag_elements:
        tag_id = tag_elem['id']
        
        # 跳过已在设备处理中使用过的位号
        if tag_id in processed_tag_ids:
            continue
        
        # 只处理在关联关系中出现过的位号
        if tag_id not in main_to_targets:
            continue
        
        # 获取关联关系
        target_ids = main_to_targets.get(tag_id, [])
        
        # 创建以位号为中心的固定尺寸窗口
        window = create_centered_window(tag_elem['pixel_bbox'], crop_size, (img_w, img_h))
        
        # 处理这个位号元件
        result = process_main_element(
            tag_id, target_ids, elements, element_map, image, window, line_width, font
        )
        
        if result is None:
            continue
        
        processed_tag_ids.add(tag_id)
        
        # 保存图片
        img_filename = f"tag_{tag_sample_idx:04d}.jpg"
        img_path = folder_output / img_filename
        result['cropped_image'].save(img_path, quality=95)
        
        # 保存JSON
        json_filename = f"tag_{tag_sample_idx:04d}.json"
        json_path = folder_output / json_filename
        sample_data = {
            'image': img_filename,
            'source_folder': folder.name,
            'crop_type': 'tag_centered',
            'main_element': result['main_element'],
            'candidate_elements': result['candidate_elements'],
            'answer': result['answer'],
        }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, ensure_ascii=False, indent=2)
        
        num_targets = len(result['answer']['target_ids'])
        sample_info = {
            'json_path': str(json_path),
            'image_path': str(img_path),
            'num_targets': num_targets,
            'crop_type': 'tag_centered',
        }
        samples.append(sample_info)
        
        tag_sample_idx += 1
    
    return samples


def preprocess_data(config: dict):
    """主预处理函数"""
    paths_cfg = config['paths']
    preprocess_cfg = config['preprocess']
    
    raw_data_dir = Path(paths_cfg['raw_data_dir'])
    output_dir = Path(paths_cfg['processed_data_dir'])
    # raw_data_dir = Path(r"关联标注审核后-2026-01-15_测试集")
    # output_dir = Path(r"test_output")
    
    print("="*60)
    print("数据预处理 - 位号中心裁剪版本")
    print("="*60)
    print(f"原始数据目录: {raw_data_dir}")
    print(f"输出目录: {output_dir}")
    print(f"位号裁剪尺寸: {preprocess_cfg.get('valve_crop_size', 1024)}x{preprocess_cfg.get('valve_crop_size', 1024)}")
    print("特点: 以位号元件为中心裁剪")
    print("      关联方向: 位号 → 符号/设备/属性标签等")
    print()
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_folders = sorted([d for d in raw_data_dir.iterdir() if d.is_dir()])
    print(f"找到 {len(all_folders)} 个数据文件夹")
    
    all_samples = []
    stats = defaultdict(int)
    crop_type_stats = defaultdict(int)
    
    for folder in tqdm(all_folders, desc="处理中"):
        samples = process_folder(folder, config, output_dir)
        all_samples.extend(samples)
        for s in samples:
            stats[f"{s['num_targets']}个关联"] += 1
            crop_type_stats[s.get('crop_type', 'valve_centered')] += 1
    
    index_path = output_dir / paths_cfg['samples_index']
    with open(index_path, 'w', encoding='utf-8') as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)
    
    print()
    print("="*60)
    print("预处理完成！")
    print("="*60)
    print(f"总样本数: {len(all_samples)}")
    print("\n裁剪类型统计:")
    for crop_type, count in sorted(crop_type_stats.items()):
        print(f"  {crop_type}: {count}")
    print("\n关联数量统计:")
    for num_type, count in sorted(stats.items()):
        print(f"  {num_type}: {count}")
    print(f"\n索引文件: {index_path}")


def main():
    parser = argparse.ArgumentParser(description='数据预处理 - 阀门中心裁剪')
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()
    
    config = load_config(args.config)
    preprocess_data(config)


if __name__ == '__main__':
    main()

