"""
推理工具函数
包含图片裁剪、数据格式转换、模型推理等功能
"""
import os
import re
import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Tuple, Optional
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


def load_model_and_processor(config: dict):
    """加载模型和处理器"""
    lora_model_path = config['model']['model_path']  # LoRA权重路径（微调后的模型）
    base_model_path = config['model']['base_model_path']  # 基础模型路径
    device = config['model'].get('device')
    
    # 验证必需参数
    if not lora_model_path:
        raise ValueError("model_path (LoRA权重路径) 不能为空")
    if not base_model_path:
        raise ValueError("base_model_path (基础模型路径) 不能为空，需要指定Qwen2.5-VL基础模型路径")
    
    print(f"正在加载模型...")
    print(f"  LoRA权重路径: {lora_model_path}")
    print(f"  基础模型路径: {base_model_path}")
    
    # 从基础模型加载processor（与evaluate.py一致）
    print(f"加载 processor from: {base_model_path}")
    processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
    processor.tokenizer.padding_side = 'left'
    
    # 加载基础模型
    print(f"加载基础模型...")
    from peft import PeftModel
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,  # 使用FP16与evaluate.py一致
        device_map="auto",
        trust_remote_code=True
    )
    
    # 加载LoRA权重（与evaluate.py一致）
    print(f"加载LoRA权重: {lora_model_path}")
    model = PeftModel.from_pretrained(base_model, lora_model_path)
    model.eval()
    
    print(f"✅ 模型加载成功")
    return model, processor


def resize_image(image: Image.Image, max_size: int = 1024) -> Image.Image:
    """等比例缩放图片"""
    w, h = image.size
    if max(w, h) <= max_size:
        return image
    if w > h:
        new_w = max_size
        new_h = int(h * max_size / w)
    else:
        new_h = max_size
        new_w = int(w * max_size / h)
    return image.resize((new_w, new_h), Image.LANCZOS)


def calculate_bbox_center(bbox: List[float]) -> Tuple[float, float]:
    """计算bbox中心点"""
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, (y1 + y2) / 2


def is_bbox_contained(outer_bbox: List[float], inner_bbox: List[float]) -> bool:
    """
    判断inner_bbox是否被outer_bbox包围
    """
    ox1, oy1, ox2, oy2 = outer_bbox
    ix1, iy1, ix2, iy2 = inner_bbox
    
    # inner完全在outer内部
    return (ox1 <= ix1 and oy1 <= iy1 and ox2 >= ix2 and oy2 >= iy2)


def calculate_bbox_distance(bbox1: List[float], bbox2: List[float]) -> float:
    """
    计算两个bbox之间的最小边界距离
    如果重叠，返回0
    """
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    
    # 计算水平和垂直方向的距离
    if x1_max < x2_min:
        dx = x2_min - x1_max
    elif x2_max < x1_min:
        dx = x1_min - x2_max
    else:
        dx = 0  # 水平方向有重叠
    
    if y1_max < y2_min:
        dy = y2_min - y1_max
    elif y2_max < y1_min:
        dy = y1_min - y2_max
    else:
        dy = 0  # 垂直方向有重叠
    
    # 返回欧式距离
    return np.sqrt(dx**2 + dy**2)


def calculate_confidence_from_geometry(main_bbox: List[float], 
                                       target_bbox: List[float],
                                       max_distance: float = 500.0) -> float:
    """
    基于几何关系计算置信度
    
    规则：
    1. 如果主体元件包围从属元件：0.85-1.0（根据包围紧密程度）
    2. 如果不包围：0.3-0.85（根据边界距离，越近越高）
    
    Args:
        main_bbox: 主体元件bbox [x1, y1, x2, y2]
        target_bbox: 从属元件bbox [x1, y1, x2, y2]
        max_distance: 最大距离阈值（超过此距离置信度为最低值）
    
    Returns:
        置信度 (0.3-1.0)
    """
    # 判断是否包围
    is_contained = is_bbox_contained(main_bbox, target_bbox)
    
    if is_contained:
        # 如果包围，基础分数高 (0.85-1.0)
        # 根据从属元件与主体元件的相对大小计算紧密度
        main_area = (main_bbox[2] - main_bbox[0]) * (main_bbox[3] - main_bbox[1])
        target_area = (target_bbox[2] - target_bbox[0]) * (target_bbox[3] - target_bbox[1])
        
        if main_area > 0:
            area_ratio = target_area / main_area
            # 面积比越小（从属元件相对越小），说明越可能是标注，置信度越高
            # 限制area_ratio在0-0.5之间，映射到0.85-1.0
            area_ratio = min(area_ratio, 0.5)
            confidence = 1.0 - (area_ratio / 0.5) * 0.15  # 0.85-1.0
        else:
            confidence = 0.9
        
        return confidence
    else:
        # 如果不包围，基于距离计算 (0.3-0.85)
        distance = calculate_bbox_distance(main_bbox, target_bbox)
        
        # 距离越近，置信度越高
        # distance=0 -> confidence=0.85
        # distance=max_distance -> confidence=0.3
        if distance >= max_distance:
            confidence = 0.3
        else:
            # 线性映射：distance从0到max_distance，confidence从0.85到0.3
            confidence = 0.85 - (distance / max_distance) * 0.55
        
        return confidence


def world_to_pixel(world_coords: List[float], 
                   world_bbox: Tuple[float, float, float, float],
                   img_size: Tuple[int, int]) -> List[int]:
    """世界坐标转像素坐标"""
    x1, y1, x2, y2 = world_coords
    w_x_min, w_y_min, w_x_max, w_y_max = world_bbox
    img_w, img_h = img_size
    
    # 计算缩放比例
    scale_x = img_w / (w_x_max - w_x_min)
    scale_y = img_h / (w_y_max - w_y_min)
    
    # 转换坐标（注意y轴反转）
    px1 = int((x1 - w_x_min) * scale_x)
    px2 = int((x2 - w_x_min) * scale_x)
    py1 = int(img_h - (y2 - w_y_min) * scale_y)
    py2 = int(img_h - (y1 - w_y_min) * scale_y)
    
    return [px1, py1, px2, py2]


def crop_image_for_element_device(image: Image.Image,
                                   main_element: Dict,
                                   candidate_elements: List[Dict],
                                   device_bbox: List[float],
                                   device_padding: int = 100,
                                   min_size: int = 1024) -> Tuple[Image.Image, Dict, List[Dict]]:
    """
    设备扩展裁剪：以设备bbox+候选元件bbox范围+padding为窗口，最小保证min_size
    """
    img_w, img_h = image.size

    # 收集设备和所有候选元件的bbox范围
    all_bboxes = [device_bbox] + [e['pixel_bbox'] for e in candidate_elements] + [main_element['pixel_bbox']]
    min_x = min(b[0] for b in all_bboxes) - device_padding
    min_y = min(b[1] for b in all_bboxes) - device_padding
    max_x = max(b[2] for b in all_bboxes) + device_padding
    max_y = max(b[3] for b in all_bboxes) + device_padding

    min_x = max(0, min_x)
    min_y = max(0, min_y)
    max_x = min(img_w, max_x)
    max_y = min(img_h, max_y)

    # 如果小于min_size则扩展
    w = max_x - min_x
    h = max_y - min_y
    if w < min_size:
        expand = (min_size - w) / 2
        min_x = max(0, min_x - expand)
        max_x = min(img_w, max_x + expand)
        if max_x - min_x < min_size:
            max_x = min(img_w, min_x + min_size)
            min_x = max(0, max_x - min_size)
    if h < min_size:
        expand = (min_size - h) / 2
        min_y = max(0, min_y - expand)
        max_y = min(img_h, max_y + expand)
        if max_y - min_y < min_size:
            max_y = min(img_h, min_y + min_size)
            min_y = max(0, max_y - min_size)

    win_x1, win_y1 = int(min_x), int(min_y)
    cropped_img = image.crop((win_x1, win_y1, int(max_x), int(max_y)))

    new_main_bbox = [
        main_element['pixel_bbox'][0] - win_x1,
        main_element['pixel_bbox'][1] - win_y1,
        main_element['pixel_bbox'][2] - win_x1,
        main_element['pixel_bbox'][3] - win_y1,
    ]
    new_candidates = []
    for cand in candidate_elements:
        cb = cand['pixel_bbox']
        new_cand = cand.copy()
        new_cand['bbox'] = [cb[0] - win_x1, cb[1] - win_y1, cb[2] - win_x1, cb[3] - win_y1]
        new_candidates.append(new_cand)

    return cropped_img, {'bbox': new_main_bbox, **main_element}, new_candidates


def crop_image_for_element(image: Image.Image,
                           main_element: Dict,
                           candidate_elements: List[Dict],
                           crop_size: int = 1024) -> Tuple[Image.Image, Dict, List[Dict]]:
    """
    以主元件为中心裁剪图片，并更新元件bbox
    """
    img_w, img_h = image.size
    
    # 计算主元件中心
    main_bbox = main_element['pixel_bbox']
    cx = (main_bbox[0] + main_bbox[2]) / 2
    cy = (main_bbox[1] + main_bbox[3]) / 2
    
    # 计算窗口左上角
    win_x1 = cx - crop_size / 2
    win_y1 = cy - crop_size / 2
    
    # 边界检查
    win_x1 = max(0, min(win_x1, img_w - crop_size))
    win_y1 = max(0, min(win_y1, img_h - crop_size))
    win_x1 = max(0, win_x1)
    win_y1 = max(0, win_y1)
    
    # 计算实际窗口尺寸
    actual_w = min(crop_size, img_w - win_x1)
    actual_h = min(crop_size, img_h - win_y1)
    
    # 裁剪图片
    crop_box = (int(win_x1), int(win_y1), int(win_x1 + actual_w), int(win_y1 + actual_h))
    cropped_img = image.crop(crop_box)
    
    # 更新主元件bbox（相对于裁剪后的图片）
    new_main_bbox = [
        main_bbox[0] - win_x1,
        main_bbox[1] - win_y1,
        main_bbox[2] - win_x1,
        main_bbox[3] - win_y1
    ]
    
    # 更新候选元件bbox
    new_candidates = []
    for cand in candidate_elements:
        cand_bbox = cand['pixel_bbox']
        new_bbox = [
            cand_bbox[0] - win_x1,
            cand_bbox[1] - win_y1,
            cand_bbox[2] - win_x1,
            cand_bbox[3] - win_y1
        ]
        new_cand = cand.copy()
        new_cand['bbox'] = new_bbox
        new_candidates.append(new_cand)
    
    return cropped_img, {'bbox': new_main_bbox, **main_element}, new_candidates


def draw_boxes_on_image(image: Image.Image,
                        main_element: Dict,
                        candidate_elements: List[Dict],
                        line_width: int = 3) -> Image.Image:
    """在图片上绘制边框"""
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except:
        font = ImageFont.load_default()
    
    # 绘制主元件（红色）
    main_bbox = main_element['bbox']
    draw.rectangle(main_bbox, outline='red', width=line_width)
    draw.text((main_bbox[0], main_bbox[1] - 20), 
             f"{main_element['simple_id']}", 
             fill='red', font=font)
    
    # 绘制候选元件（黄色）
    for cand in candidate_elements:
        cand_bbox = cand['bbox']
        draw.rectangle(cand_bbox, outline='yellow', width=line_width)
        draw.text((cand_bbox[0], cand_bbox[1] - 20), 
                 f"{cand['simple_id']}", 
                 fill='yellow', font=font)
    
    return img_draw


def build_prompt_for_inference(main_element: Dict, 
                               candidate_elements: List[Dict],
                               category_map: Dict[str, str]) -> str:
    """
    构建推理提示词（与训练时完全一致）
    使用dataset.py中的build_prompt格式
    """
    elements_info = []
    
    # 主元件（红色框）
    main_cat_en = category_map.get(main_element['category'], main_element['category'].lower())
    elements_info.append({
        'id': main_element['simple_id'],
        'category': main_cat_en,
        'color': 'red',
        'bbox': main_element.get('bbox', [0, 0, 0, 0])
    })
    
    # 候选元件（黄色框）
    for elem in candidate_elements:
        elem_cat_en = category_map.get(elem['category'], elem['category'].lower())
        elements_info.append({
            'id': elem['simple_id'],
            'category': elem_cat_en,
            'color': 'yellow',
            'bbox': elem.get('bbox', [0, 0, 0, 0])
        })
    
    elements_json = json.dumps(elements_info, ensure_ascii=False, indent=2)
    
    # 使用设备/符号为主体的提示词格式
    prompt = f"""你是工程CAD图纸中的【元件关联识别助手】。

任务：
判断【红色框元件】是否与【黄色框元件】存在工程上的直接关联关系，
设备、符号在语义上属于红色框元件,并从黄色框中选出 1～2 个与红色框最相关的元件。

关联关系的判定标准（满足其一即可）：
1. 设备序号/位号/属性标签明显标注或指向红色框元件
2. 空间位置上明显围绕、贴近或对齐红色框，且具有工程含义
3. 工程常识上属于"symbol-tag / symbol-label / device-device_number / device-tag"的组合

不属于关联的情况（必须排除）：
- 仅空间接近但不符合工程常识
- 黄色框元件属于其他设备或符号
- 无法确定工程指向关系

输入说明：
- 红色框：查询主体元件（ID1）
- 黄色框：候选关联元件（多个）
- 框左上角绿色数字为元件ID
- bbox坐标：以图中的左上角为原点的坐标系

输入数据格式：
{elements_json}

输出要求（严格遵守）：
- 如果存在关联：只输出黄色框的ID数字，按相关性从高到低排序，多个用英文逗号分隔
- 如果关联**2个同类别元件**只选最相关1个
- 如果不存在关联：只输出单词 NONE（全大写，无引号）
- 禁止输出任何其他文字、解释、标点或空格
"""

    return prompt


def parse_model_output(output_text: str) -> List[int]:
    """
    解析模型输出（与evaluate.py中的parse_model_output一致）

    支持格式：
    - "3"
    - "2, 3"
    - "2,3"
    - "NONE" / "none" / "无" -> []
    """
    output = output_text.strip()

    # 显式处理无关联的情况（与evaluate.py保持一致）
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


def convert_to_association_format(main_element: Dict,
                                  target_ids: List[int],
                                  candidate_elements: List[Dict],
                                  default_conf: float = 0.9,
                                  use_geometry_confidence: bool = True) -> List[Dict]:
    """
    将模型输出转换为接口返回格式
    过滤不合理的关联：
    1. 主体元件之间不能关联（设备-设备、设备-符号、符号-符号等）
    2. 设备-属性标签不能关联
    3. 符号-设备序号不能关联
    """
    associations = []

    # 定义主体元件类别
    main_categories = {'设备', '符号', '仪表'}

    # 构建候选元件ID映射
    cand_map = {c['simple_id']: c for c in candidate_elements}

    filtered_count = 0  # 过滤计数

    for target_id in target_ids:
        if target_id not in cand_map:
            continue

        target_elem = cand_map[target_id]

        # 使用原始类别名称进行判断
        main_cat = main_element.get('original_category') or main_element.get('category', '')
        target_cat = target_elem.get('original_category') or target_elem.get('category', '')

        # 标准化类别名称（处理可能的subclass字段）
        if not main_cat and 'subclass' in main_element:
            main_cat = main_element['subclass']
        if not target_cat and 'subclass' in target_elem:
            target_cat = target_elem['subclass']

        # 过滤规则1：涉及仪表的关联不计入结果
        if main_cat == '仪表' or target_cat == '仪表':
            filtered_count += 1
            continue

        # 过滤规则2：主体元件之间不能关联（设备-设备、符号-符号等）
        if main_cat in main_categories and target_cat in main_categories:
            filtered_count += 1
            continue

        # 过滤规则3：设备-属性标签不能关联
        if main_cat == '设备' and target_cat == '属性标签':
            filtered_count += 1
            continue

        # 过滤规则4：符号-设备序号不能关联
        if main_cat == '符号' and target_cat == '设备序号':
            filtered_count += 1
            continue

        # 确定关联类型
        if main_cat == '设备' and target_cat == '设备序号':
            rel_type = 'device-number'
        elif main_cat == '设备' and target_cat == '位号':
            rel_type = 'device-tag'
        elif main_cat == '符号' and target_cat == '位号':
            rel_type = 'symbol-tag'
        elif main_cat == '符号' and target_cat == '属性标签':
            rel_type = 'symbol-attr'
        elif main_cat == '仪表' and target_cat == '位号':
            rel_type = 'instrument-tag'
        else:
            # 默认类型（小写，用连字符）
            main_cat_lower = main_cat.lower() if main_cat else 'unknown'
            target_cat_lower = target_cat.lower() if target_cat else 'unknown'
            rel_type = f"{main_cat_lower}-{target_cat_lower}"

        # 计算置信度
        if use_geometry_confidence:
            # 基于几何关系计算置信度
            main_bbox = main_element.get('pixel_bbox', [0, 0, 0, 0])
            target_bbox = target_elem.get('pixel_bbox', [0, 0, 0, 0])
            confidence = calculate_confidence_from_geometry(main_bbox, target_bbox)
        else:
            # 使用默认置信度
            confidence = default_conf

        associations.append({
            'source': main_element.get('id', ''),
            'target': target_elem.get('id', ''),
            'type': rel_type,
            'conf': round(confidence, 3)  # 保留3位小数
        })

    return associations


def select_candidates(main_element: Dict,
                     all_elements: List[Dict],
                     max_candidates: int = 10) -> List[Dict]:
    """
    选择候选元件（基于距离）

    候选元件包括所有其他元件（位号、设备序号、属性标签等）
    但需要排除：
    - 端口
    - V、P、Ø开头的属性标签
    - 小类名称中不带"阀"字的符号
    - 仪表本身
    - bbox完全在某个仪表内部的元件
    """
    main_cx, main_cy = calculate_bbox_center(main_element['pixel_bbox'])

    # 预先收集所有仪表的bbox，用于判断元件是否在仪表内部
    instrument_bboxes = [
        e['pixel_bbox'] for e in all_elements if e.get('category') == '仪表'
    ]

    def is_inside_instrument(bbox):
        """判断元件是否完全在某个仪表内部"""
        for ib in instrument_bboxes:
            if bbox[0] >= ib[0] and bbox[1] >= ib[1] and bbox[2] <= ib[2] and bbox[3] <= ib[3]:
                return True
        return False

    # 计算所有其他元件到主元件的距离
    distances = []
    for elem in all_elements:
        if elem['id'] == main_element['id']:
            continue

        # 过滤规则1: 跳过端口
        if elem.get('category') == '端口':
            continue

        # 过滤规则2: 跳过V、P、Ø开头的属性标签
        if elem.get('category') == '属性标签':
            content = elem.get('content', '').strip()
            if content and (content.startswith('V') or content.startswith('P') or content.startswith('Ø')):
                continue

        # 过滤规则3: 符号类必须在subclass中包含"阀"字
        if elem.get('category') == '符号':
            subclass = elem.get('subclass', '').strip()
            if '阀' not in subclass:
                continue

        # 过滤规则4: 跳过仪表本身
        if elem.get('category') == '仪表':
            continue

        # 过滤规则5: 跳过完全在仪表内部的元件
        if is_inside_instrument(elem['pixel_bbox']):
            continue

        elem_cx, elem_cy = calculate_bbox_center(elem['pixel_bbox'])
        dist = np.sqrt((elem_cx - main_cx)**2 + (elem_cy - main_cy)**2)
        distances.append((dist, elem))
    
    # 按距离排序
    distances.sort(key=lambda x: x[0])
    
    # 取前N个
    candidates = [elem for dist, elem in distances[:max_candidates]]
    
    return candidates


def transform_tag_associations(associations: List[Dict]) -> List[Dict]:
    """
    将位号为主体的关联关系转换为符号/设备为主体的关联关系

    转换规则：
    - 位号→符号 + 位号→属性标签 => 符号→属性标签
    - 位号→设备 + 位号→设备序号 => 设备→设备序号
    - 保留位号→符号、位号→设备的关联

    Args:
        associations: 原始关联关系列表

    Returns:
        转换后的关联关系列表
    """
    # 按位号（source）分组
    tag_groups = {}
    for assoc in associations:
        source = assoc['source']
        if source not in tag_groups:
            tag_groups[source] = []
        tag_groups[source].append(assoc)

    result = []

    for _, assocs in tag_groups.items():
        # 分类关联对象
        symbol_assoc = None
        device_assoc = None
        attr_assocs = []
        device_num_assocs = []
        other_assocs = []

        for assoc in assocs:
            rel_type = assoc['type']
            if rel_type == 'tag-symbol' or rel_type == '位号-符号':
                symbol_assoc = assoc
            elif rel_type == 'tag-device' or rel_type == '位号-设备':
                device_assoc = assoc
            elif rel_type == 'tag-label' or rel_type == '位号-属性标签':
                attr_assocs.append(assoc)
            elif rel_type == 'tag-device_number' or rel_type == '位号-设备序号':
                device_num_assocs.append(assoc)
            else:
                other_assocs.append(assoc)

        # 转换规则1: 位号→符号 + 位号→属性标签 => 符号→属性标签
        if symbol_assoc and attr_assocs:
            # 保留位号→符号
            result.append(symbol_assoc)
            # 转换为符号→属性标签
            for attr_assoc in attr_assocs:
                result.append({
                    'source': symbol_assoc['target'],  # 符号ID
                    'target': attr_assoc['target'],     # 属性标签ID
                    'type': 'symbol-attr',
                    'conf': attr_assoc['conf']
                })
        elif symbol_assoc:
            # 只有位号→符号，保留
            result.append(symbol_assoc)
        elif attr_assocs:
            # 只有位号→属性标签，保留（虽然不太可能）
            result.extend(attr_assocs)

        # 转换规则2: 位号→设备 + 位号→设备序号 => 设备→设备序号
        if device_assoc and device_num_assocs:
            # 保留位号→设备
            result.append(device_assoc)
            # 转换为设备→设备序号
            for num_assoc in device_num_assocs:
                result.append({
                    'source': device_assoc['target'],  # 设备ID
                    'target': num_assoc['target'],     # 设备序号ID
                    'type': 'device-number',
                    'conf': num_assoc['conf']
                })
        elif device_assoc:
            # 只有位号→设备，保留
            result.append(device_assoc)
        elif device_num_assocs:
            # 只有位号→设备序号，保留（虽然不太可能）
            result.extend(device_num_assocs)

        # 保留其他关联
        result.extend(other_assocs)

    return result


