"""
VLM 图构建服务
替换原有的 calculate_associations 接口。
输入：图片 + 元件分类结果 + 原始图元(含线条) + 世界坐标
输出：与 graph-02.json 格式一致的图结构（nodes + edges）
"""
import base64
import json
import math
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml
from flask import Flask, request, jsonify
from openai import OpenAI
from PIL import Image

# ==================== 导入配置 ====================
config_path = Path(__file__).parent / 'config.yaml'
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

app = Flask(__name__)

# ==================== vLLM 客户端 ====================
vllm_base_url = config['model']['vllm_base_url']
vllm_model_name = config['model']['vllm_model_name']
vllm_client = OpenAI(api_key="EMPTY", base_url=vllm_base_url)

print(f"✅ VLM 图构建服务初始化")
print(f"   - vLLM地址: {vllm_base_url}")
print(f"   - 模型名称: {vllm_model_name}")

# ==================== SYSTEM_PROMPT ====================
SYSTEM_PROMPT = (
    "你是一个用于工艺系统图连接关系抽取的视觉结构化助手。"
    "你会同时看到一张子图图片，以及用户提供的图元清单。"
    "图元清单只包含两类输入：bbox 元件和 line 线段。"
    "每个 bbox 元件通常带有 name、type 和 bbox=[x1,y1,x2,y2]；"
    "每个 line 通常带有 name、type、point1 和 point2。"
    "这里要求输出的 connections，本质上就是输入的 bbox 元件和 line 线段这些图元之间的连接关系。"
    "你的任务是根据图片和给定的 bbox、line 图元，预测当前子图中的 connections。"
    "你的任务不是复述图片内容，也不是解释图纸，而是根据图片中的局部拓扑关系、"
    "图元坐标、图元类型、线段位置与连通方式，推断并输出当前子图内真实存在的 connections。"
    "需要特别注意：输入中的 name 只是图元标识符，未必与图中可见文字标注完全一致；"
    "也就是说，图里的名称文字可能和 bbox 或 line 提供的 name 不同，但对应的几何位置和坐标是一致的。"
    "因此判断连接关系时，应优先依据图像里的实际连线、局部拓扑结构和输入图元坐标对齐关系，而不是依赖图中文字与输入 name 的字面一致。"
    "输出时必须严格遵守以下规则："
    "第一，只输出连接结果，不输出分析过程。"
    "第二，只基于当前子图中的图像内容和给定图元作答。"
    "第三，只输出当前输入里能够被局部图像和给定图元支持的 connections。"
    "第四，connections 中出现的 Name 和 Connectto 都必须引用输入中出现过的 name，且 Type 应与输入中对应 name 的 type 一致。"
    "第五，如果一个组件没有足够证据确认连接关系，不要猜测，不要补充不存在的连接。"
    "第六，优先学习几何邻接、线段端点落点、线段穿过或接触元件边界、局部拓扑连续性等模式。"
    "第七，输出必须是 JSON 对象，且只包含一个键 connections。"
    "第八，最终输出格式应为 {\"connections\": [{\"Name\": \"输入中的name\", \"Type\": \"输入中对应的type\", \"Connectto\": [\"输入中的name1\", \"输入中的name2\"]}]}；如果没有可靠连接，输出 {\"connections\": []}。"
)


# ==================== 坐标变换 ====================
def world_to_pixel(
    world_coords: List[float],
    world_bbox: Tuple[float, float, float, float],
    img_size: Tuple[int, int],
) -> List[int]:
    """世界坐标转像素坐标"""
    x1, y1, x2, y2 = world_coords
    w_x_min, w_y_min, w_x_max, w_y_max = world_bbox
    img_w, img_h = img_size

    scale_x = img_w / (w_x_max - w_x_min) if w_x_max != w_x_min else 1
    scale_y = img_h / (w_y_max - w_y_min) if w_y_max != w_y_min else 1

    px1 = int((x1 - w_x_min) * scale_x)
    px2 = int((x2 - w_x_min) * scale_x)
    py1 = int(img_h - (y2 - w_y_min) * scale_y)
    py2 = int(img_h - (y1 - w_y_min) * scale_y)

    return [px1, py1, px2, py2]


def point_world_to_pixel(
    pt: List[float],
    world_bbox: Tuple[float, float, float, float],
    img_size: Tuple[int, int],
) -> List[int]:
    """单个点坐标转换"""
    w_x_min, w_y_min, w_x_max, w_y_max = world_bbox
    img_w, img_h = img_size
    scale_x = img_w / (w_x_max - w_x_min) if w_x_max != w_x_min else 1
    scale_y = img_h / (w_y_max - w_y_min) if w_y_max != w_y_min else 1
    px = int((pt[0] - w_x_min) * scale_x)
    py = int(img_h - (pt[1] - w_y_min) * scale_y)
    return [px, py]


# ==================== 图构建工具 ====================
def build_vlm_bbox_input(classifications: List[Dict]) -> List[Dict]:
    """将分类结果转为 VLM 输入的 bbox 格式"""
    bbox_items = []
    for ele in classifications:
        pixel_bbox = ele.get('pixel_bbox')
        if not pixel_bbox:
            continue
        type_str = ele.get('category', '未知')
        subclass = ele.get('subclass', '')
        if subclass and subclass != type_str:
            type_str = f"{type_str}-{subclass}"
        bbox_items.append({
            "name": ele['id'],
            "type": type_str,
            "bbox": pixel_bbox,
        })
    return bbox_items


def build_vlm_line_input(segments: List[Dict], world_bbox, img_size) -> List[Dict]:
    """将标准化线段转为 VLM 输入的 line 格式"""
    line_items = []
    for seg in segments:
        points = seg.get("points", [])
        if len(points) < 2:
            continue
        seg_id = seg.get("id", f"S_{seg.get('idx', 0):05d}")
        # 端点已在 world 坐标中，转 pixel
        p1 = point_world_to_pixel(points[0], world_bbox, img_size)
        p2 = point_world_to_pixel(points[-1], world_bbox, img_size)
        line_items.append({
            "name": seg_id,
            "type": "line-管道",
            "point1": p1,
            "point2": p2,
        })
    return line_items


def image_to_data_url(image: Image.Image, image_format: str = "PNG") -> str:
    """将 PIL 图片转为 base64 数据 URL"""
    buffer = BytesIO()
    image.save(buffer, format=image_format)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    mime = f"image/{image_format.lower()}"
    return f"data:{mime};base64,{encoded}"


def call_vlm_tile(
    tile_image: Image.Image,
    tile_bbox_items: List[Dict],
    tile_line_items: List[Dict],
) -> Dict:
    """
    对单个 tile 调用 vLLM 推理。
    返回: {"connections": [...]} 或 {"connections": [], "parse_error": "..."}
    """
    # 构建 user 文本
    payload = {"bbox": tile_bbox_items, "line": tile_line_items}
    text_content = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text_content},
                {"type": "image_url", "image_url": {"url": image_to_data_url(tile_image)}},
            ],
        },
    ]

    try:
        response = vllm_client.chat.completions.create(
            model=vllm_model_name,
            messages=messages,
            max_tokens=8192,
            temperature=0.0,
            top_p=1.0,
        )
        output_text = response.choices[0].message.content
    except Exception as e:
        return {"connections": [], "parse_error": str(e)}

    # 解析 JSON
    try:
        # 尝试提取 JSON 对象
        content = output_text.strip()
        start = content.find('{')
        end = content.rfind('}')
        if start != -1 and end != -1:
            content = content[start:end + 1]
        parsed = json.loads(content)
        connections = parsed.get("connections", [])
        return {"connections": connections, "parse_error": None}
    except (ValueError, json.JSONDecodeError) as e:
        return {"connections": [], "parse_error": str(e), "raw": output_text}


def extract_json_object(text: str) -> str:
    """从文本中提取第一个 JSON 对象"""
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]
    return text


def segment_bbox(points: List[List[float]]) -> List[float]:
    """计算线段包围盒"""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def line_intersection(
    p1: Tuple[float, float], p2: Tuple[float, float],
    p3: Tuple[float, float], p4: Tuple[float, float],
) -> Optional[Tuple[float, float]]:
    """计算两条线段的交点（若不相交返回 None）"""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10:
        return None

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

    if 0 <= t <= 1 and 0 <= u <= 1:
        ix = x1 + t * (x2 - x1)
        iy = y1 + t * (y2 - y1)
        return (ix, iy)
    return None


def point_to_segment_closest(
    pt: Tuple[float, float],
    seg_p1: Tuple[float, float],
    seg_p2: Tuple[float, float],
) -> Tuple[float, float]:
    """计算点到线段的最接近点"""
    x, y = pt
    x1, y1 = seg_p1
    x2, y2 = seg_p2

    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0:
        return (x1, y1)

    t = ((x - x1) * dx + (y - y1) * dy) / (dx * dx + dy * dy)
    t = max(0, min(1, t))
    return (x1 + t * dx, y1 + t * dy)


def compute_connect_point(
    ele_id1: str, ele_id2: str,
    segment_id_to_points: Dict[str, List[List[float]]],
    pe_id_to_bbox: Dict[str, List[float]],
) -> Optional[List[float]]:
    """
    计算连接点坐标。
    - S↔S：两条线段交点
    - PE↔S：线段上最接近元件中心的位置
    """
    # 确定哪个是线段，哪个是元件
    seg_id = None
    pe_id = None
    if ele_id1.startswith('S_') and ele_id2.startswith('S_'):
        # S↔S
        pts1 = segment_id_to_points.get(ele_id1)
        pts2 = segment_id_to_points.get(ele_id2)
        if pts1 and pts2 and len(pts1) >= 2 and len(pts2) >= 2:
            inter = line_intersection(
                tuple(pts1[0]), tuple(pts1[-1]),
                tuple(pts2[0]), tuple(pts2[-1]),
            )
            if inter:
                return [round(inter[0], 2), round(inter[1], 2)]
        return None

    if ele_id1.startswith('S_'):
        seg_id = ele_id1
        pe_id = ele_id2
    elif ele_id2.startswith('S_'):
        seg_id = ele_id2
        pe_id = ele_id1

    if seg_id and pe_id:
        pts = segment_id_to_points.get(seg_id)
        pe_bbox = pe_id_to_bbox.get(pe_id)
        if pts and pe_bbox and len(pts) >= 2:
            # 使用元件中心
            cx = (pe_bbox[0] + pe_bbox[2]) / 2
            cy = (pe_bbox[1] + pe_bbox[3]) / 2
            cp = point_to_segment_closest((cx, cy), tuple(pts[0]), tuple(pts[-1]))
            return [round(cp[0], 2), round(cp[1], 2)]

    return None


# ==================== 主构建函数 ====================
def build_graph_from_vlm(
    classifications: List[Dict],
    segments: List[Dict],
    vlm_connections: List[Dict],
) -> Dict:
    """
    从 VLM 输出构建 graph-02.json 格式的图结构。

    Args:
        classifications: 检测到的元件列表（含 id, category, x1/y1/x2/y2）
        segments: 标准化线段列表（含 id, points, idx）
        vlm_connections: VLM 输出的 connections 列表

    Returns:
        {"nodes": [...], "edges": [...]}
    """
    # 构建 PE 节点
    pe_id_to_bbox = {}
    node_set = {}  # ele_id -> node dict

    for ele in classifications:
        ele_id = ele['id']
        bbox = [ele.get('x1', 0), ele.get('y1', 0), ele.get('x2', 0), ele.get('y2', 0)]
        pe_id_to_bbox[ele_id] = bbox

        label = ele.get('category', '未知')
        subclass = ele.get('subclass', '')
        cate = label
        if subclass and subclass != label:
            cate = f"{label}-{subclass}"

        node_set[ele_id] = {
            "label": label,
            "cate": cate,
            "ele_id": ele_id,
            "bbox": bbox,
        }

    # 构建 S 节点
    segment_id_to_points = {}
    for seg in segments:
        seg_id = seg.get('id', f"S_{seg.get('idx', 0):05d}")
        points = seg.get('points', [])
        segment_id_to_points[seg_id] = points
        bbox = segment_bbox(points)
        node_set[seg_id] = {
            "label": "线条",
            "cate": "线条",
            "ele_id": seg_id,
            "bbox": bbox,
        }

    # 构建边
    edges = []
    edge_keys = set()
    con_idx = 0

    all_names = set(node_set.keys())

    for conn in vlm_connections:
        source = conn.get("Name", "")
        targets = conn.get("Connectto", [])
        if not isinstance(targets, list):
            targets = [targets]

        # 过滤掉不存在于当前图中的 name
        targets = [t for t in targets if t in all_names and t != source]
        if not targets:
            continue

        for target in targets:
            # 判断类型
            if source.startswith("S_") or target.startswith("S_"):
                conn_type = "连接"
            else:
                conn_type = "从属"

            edge_key = (source, target)
            if edge_key in edge_keys:
                continue
            edge_keys.add(edge_key)

            edge = {
                "con_id": f"PC{con_idx:05d}",
                "connection": conn_type,
                "ele_id1": source,
                "ele_id2": target,
            }

            if conn_type == "连接":
                cp = compute_connect_point(
                    source, target,
                    segment_id_to_points, pe_id_to_bbox,
                )
                if cp:
                    edge["attributes"] = {"connect_point": cp}

            edges.append(edge)
            con_idx += 1

    nodes = sorted(node_set.values(), key=lambda n: n["ele_id"])

    return {"nodes": nodes, "edges": edges}


# ==================== 瓦片处理 ====================
class Tile:
    """瓦片定义"""
    def __init__(self, x: int, y: int, w: int, h: int):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.x2 = x + w
        self.y2 = y + h

    def id_str(self) -> str:
        return f"{self.x}_{self.y}_{self.w}_{self.h}"


def build_tiles(img_w: int, img_h: int, tile_size: int = 1024, overlap: int = 50) -> List[Tile]:
    """将大图划分为瓦片"""
    tiles = []
    stride = tile_size - overlap
    for y in range(0, img_h, stride):
        for x in range(0, img_w, stride):
            tw = min(tile_size, img_w - x)
            th = min(tile_size, img_h - y)
            if tw > overlap and th > overlap:
                tiles.append(Tile(x, y, tw, th))
    return tiles


def filter_items_for_tile(
    bbox_items: List[Dict],
    line_items: List[Dict],
    tile: Tile,
    keep_ratio: float = 0.3,
) -> Tuple[List[Dict], List[Dict], List[str]]:
    """
    过滤落入 tile 范围内的 bbox 和 line。
    返回: (tile_bbox, tile_line, tile_names)
    """
    tx1, ty1, tx2, ty2 = tile.x, tile.y, tile.x + tile.w, tile.y + tile.h

    tile_bbox = []
    bbox_names = []
    for item in bbox_items:
        b = item["bbox"]
        # 计算重叠面积比例
        ix1 = max(b[0], tx1)
        iy1 = max(b[1], ty1)
        ix2 = min(b[2], tx2)
        iy2 = min(b[3], ty2)
        if ix2 > ix1 and iy2 > iy1:
            inter_area = (ix2 - ix1) * (iy2 - iy1)
            bbox_area = (b[2] - b[0]) * (b[3] - b[1])
            if bbox_area > 0 and inter_area / bbox_area >= keep_ratio:
                tile_bbox.append(item)
                bbox_names.append(item["name"])

    tile_line = []
    line_names = []
    for item in line_items:
        p1, p2 = item["point1"], item["point2"]
        # 检查线段中点是否在 tile 内
        mx = (p1[0] + p2[0]) / 2
        my = (p1[1] + p2[1]) / 2
        if tx1 <= mx <= tx2 and ty1 <= my <= ty2:
            tile_line.append(item)
            line_names.append(item["name"])
        else:
            # 检查线段是否有端点在 tile 内
            p1_in = tx1 <= p1[0] <= tx2 and ty1 <= p1[1] <= ty2
            p2_in = tx1 <= p2[0] <= tx2 and ty1 <= p2[1] <= ty2
            if p1_in or p2_in:
                tile_line.append(item)
                line_names.append(item["name"])

    all_names = set(bbox_names) | set(line_names)
    # 对 bbox 做坐标局部化
    for item in tile_bbox:
        item["bbox"] = [item["bbox"][0] - tile.x, item["bbox"][1] - tile.y,
                        item["bbox"][2] - tile.x, item["bbox"][3] - tile.y]
    for item in tile_line:
        item["point1"] = [item["point1"][0] - tile.x, item["point1"][1] - tile.y]
        item["point2"] = [item["point2"][0] - tile.x, item["point2"][1] - tile.y]

    return tile_bbox, tile_line, list(all_names)


def merge_connections(all_connections: List[Dict]) -> List[Dict]:
    """
    合并多 tile 的 connections，去重。
    返回合并后的 connections 列表。
    """
    merged = {}  # Name -> {Type, Connectto: set}
    for conn in all_connections:
        name = conn.get("Name")
        if not name:
            continue
        conn_type = conn.get("Type")
        targets = conn.get("Connectto", [])
        if not isinstance(targets, list):
            targets = [targets]

        if name not in merged:
            merged[name] = {"Type": conn_type, "Connectto": set()}
        else:
            if conn_type and not merged[name]["Type"]:
                merged[name]["Type"] = conn_type
        merged[name]["Connectto"].update(targets)

    result = []
    for name, data in merged.items():
        result.append({
            "Name": name,
            "Type": data["Type"],
            "Connectto": sorted(data["Connectto"]),
        })
    return result


# ==================== API 路由 ====================
@app.route('/health', methods=['GET'])
def health():
    """健康检查"""
    try:
        models = vllm_client.models.list()
        model_ids = [m.id for m in models.data]
        return jsonify({
            'success': True,
            'service': 'VLM Graph Service',
            'vllm_models': model_ids,
            'status': 'ready'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e), 'status': 'vLLM unavailable'}), 503


@app.route('/calculate_associations', methods=['POST'])
def calculate_associations():
    """
    计算关联关系并构建图。
    接收图片、元件分类、原始图元，返回完整的 graph 数据。

    请求参数 (multipart/form-data):
        image: 图片文件
        classifications: JSON 字符串，元件检测结果 [{id, category, x1,y1,x2,y2, ...}]
        element_lists: JSON 字符串，原始 DWG 图元列表
        x_min, y_min, x_max, y_max: 世界坐标系范围
    """
    t_start = time.time()

    try:
        # ========== 1. 解析请求 ==========
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': '缺少 image'}), 400

        image_file = request.files['image']
        classifications_str = request.form.get('classifications', '')
        element_lists_str = request.form.get('element_lists', '')

        if not classifications_str:
            return jsonify({'success': False, 'error': '缺少 classifications'}), 400
        if not element_lists_str:
            return jsonify({'success': False, 'error': '缺少 element_lists'}), 400

        try:
            classifications = json.loads(classifications_str)
        except json.JSONDecodeError as e:
            return jsonify({'success': False, 'error': f'classifications JSON 解析失败: {e}'}), 400

        try:
            element_lists = json.loads(element_lists_str)
        except json.JSONDecodeError as e:
            return jsonify({'success': False, 'error': f'element_lists JSON 解析失败: {e}'}), 400

        try:
            x_min = float(request.form.get('x_min', 0))
            y_min = float(request.form.get('y_min', 0))
            x_max = float(request.form.get('x_max', 0))
            y_max = float(request.form.get('y_max', 0))
        except ValueError as e:
            return jsonify({'success': False, 'error': f'坐标参数错误: {e}'}), 400

        world_bbox = (x_min, y_min, x_max, y_max)

        # ========== 2. 加载图片 ==========
        image_file.seek(0)
        try:
            image = Image.open(image_file).convert('RGB')
        except Exception as e:
            return jsonify({'success': False, 'error': f'图片加载失败: {e}'}), 400
        img_size = image.size

        print(f"📷 图片尺寸: {img_size}, 元件数: {len(classifications)}, 图元数: {len(element_lists)}")
        t1 = time.time()

        # ========== 3. 提取线条 (复用 build_graph.py) ==========
        from line_extractor import extract_lines

        # 有效区域 = 所有元件的 bbox 外扩
        if classifications:
            xs = [c.get('x1', 0) for c in classifications] + [c.get('x2', 0) for c in classifications]
            ys = [c.get('y1', 0) for c in classifications] + [c.get('y2', 0) for c in classifications]
            margin = 500.0
            region = (min(xs) - margin, min(ys) - margin, max(xs) + margin, max(ys) + margin)
        else:
            region = world_bbox

        segments = extract_lines(element_lists, region=region, components=classifications)
        # 为每个线段分配 S_ID
        for i, seg in enumerate(segments):
            if 'id' not in seg or not seg['id']:
                seg['id'] = f"S_{i:05d}"

        print(f"📏 提取线条: {len(segments)} 条, 耗时 {time.time() - t1:.2f}s")
        t2 = time.time()

        # ========== 4. 坐标转换 (world -> pixel) ==========
        for ele in classifications:
            pixel_bbox = world_to_pixel(
                [ele.get('x1', 0), ele.get('y1', 0), ele.get('x2', 0), ele.get('y2', 0)],
                world_bbox, img_size,
            )
            ele['pixel_bbox'] = pixel_bbox

        # ========== 5. 构建 VLM 输入 ==========
        bbox_items_all = build_vlm_bbox_input(classifications)
        line_items_all = build_vlm_line_input(segments, world_bbox, img_size)

        print(f"🔤 VLM 输入: bbox={len(bbox_items_all)}, line={len(line_items_all)}, 耗时 {time.time() - t2:.2f}s")

        # ========== 6. 判断是否需要切分 ==========
        MAX_DIRECT_SIZE = 2048  # 直接推理的最大尺寸
        TILE_SIZE = 1024
        TILE_OVERLAP = 50

        if img_size[0] > MAX_DIRECT_SIZE or img_size[1] > MAX_DIRECT_SIZE:
            # 瓦片推理
            tiles = build_tiles(img_size[0], img_size[1], TILE_SIZE, TILE_OVERLAP)
            print(f"🧩 大图切分: {len(tiles)} 个 tile")

            all_connections = []
            tile_errors = []

            with ThreadPoolExecutor(max_workers=8) as executor:
                future_to_tile = {}
                for tile in tiles:
                    tile_bbox, tile_line, tile_names = filter_items_for_tile(
                        [dict(b) for b in bbox_items_all],
                        [dict(l) for l in line_items_all],
                        tile,
                    )
                    if not tile_bbox and not tile_line:
                        continue
                    if not tile_names:
                        continue

                    tile_img = image.crop((tile.x, tile.y, tile.x2, tile.y2))
                    future = executor.submit(
                        call_vlm_tile, tile_img, tile_bbox, tile_line,
                    )
                    future_to_tile[future] = tile

                for future in as_completed(future_to_tile):
                    tile = future_to_tile[future]
                    try:
                        result = future.result()
                        if result.get("parse_error"):
                            tile_errors.append({"tile": tile.id_str(), "error": result["parse_error"]})
                            print(f"  ⚠️  tile {tile.id_str()} 解析失败: {result['parse_error']}")
                        else:
                            all_connections.extend(result.get("connections", []))
                            print(f"  ✅ tile {tile.id_str()} -> {len(result.get('connections', []))} 条连接")
                    except Exception as e:
                        tile_errors.append({"tile": tile.id_str(), "error": str(e)})
                        print(f"  ❌ tile {tile.id_str()} 异常: {e}")

            # 合并多 tile 结果
            merged_connections = merge_connections(all_connections)
            print(f"🔗 VLM 总连接数: {len(all_connections)} (合并后: {len(merged_connections)})")
            if tile_errors:
                print(f"⚠️  {len(tile_errors)} 个 tile 存在错误")

        else:
            # 直接推理（无需切分）
            print(f"🧠 单次推理（无需切分）")
            result = call_vlm_tile(image, bbox_items_all, line_items_all)
            if result.get("parse_error"):
                print(f"⚠️  推理异常: {result['parse_error']}")
                merged_connections = []
            else:
                merged_connections = result.get("connections", [])
                print(f"🔗 VLM 输出: {len(merged_connections)} 条连接")

        t3 = time.time()
        print(f"⏱ VLM 推理耗时: {time.time() - t2:.2f}s")

        # ========== 7. 构建图 ==========
        graph = build_graph_from_vlm(
            classifications,
            segments,
            merged_connections,
        )
        print(f"📊 图构建完成: {len(graph['nodes'])} nodes, {len(graph['edges'])} edges, 总耗时 {time.time() - t_start:.2f}s")

        # 从 graph edges 中提取传统 associations 格式（兼容下游 save_step2_result）
        vlm_associations = []
        classifications_by_id = {c['id']: c for c in classifications}
        for edge in graph.get("edges", []):
            if edge.get("connection") == "从属":
                source_id = edge["ele_id1"]
                target_id = edge["ele_id2"]
                # 确定关联类型（基于元件类别）
                src = classifications_by_id.get(source_id, {})
                tgt = classifications_by_id.get(target_id, {})
                src_label = src.get("category", "")
                tgt_label = tgt.get("category", "")
                if src_label == "设备" and tgt_label == "设备序号":
                    rel_type = "device-number"
                elif src_label == "设备" and tgt_label == "位号":
                    rel_type = "device-tag"
                elif src_label == "符号" and tgt_label == "位号":
                    rel_type = "symbol-tag"
                elif src_label == "符号" and tgt_label == "属性标签":
                    rel_type = "symbol-attr"
                else:
                    rel_type = f"{src_label}-{tgt_label}" if src_label and tgt_label else "unknown"
                vlm_associations.append({
                    "source": source_id,
                    "target": target_id,
                    "type": rel_type,
                    "conf": 1.0,
                })

        return jsonify({
            'success': True,
            'graph': graph,
            'associations': vlm_associations,
            'stats': {
                'total_elements': len(classifications),
                'total_segments': len(segments),
                'total_connections': len(merged_connections),
                'nodes': len(graph['nodes']),
                'edges': len(graph['edges']),
                'duration': round(time.time() - t_start, 2),
            },
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


# ==================== 启动 ====================
if __name__ == '__main__':
    flask_config = config['flask']
    host = flask_config['host']
    port = flask_config['port']
    debug = flask_config.get('debug', False)
    max_content_mb = flask_config.get('max_content_length', 100)

    print(f"\n{'='*60}")
    print(f"🚀 启动 VLM 图构建服务")
    print(f"{'='*60}")
    print(f"服务地址: http://{host}:{port}")
    print(f"健康检查: http://{host}:{port}/health")
    print(f"推理接口: http://{host}:{port}/calculate_associations")
    print(f"{'='*60}\n")

    app.run(host=host, port=port, debug=debug)
