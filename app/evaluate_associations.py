"""
关联关系识别端到端评估脚本
=================================
流程：
  1. 从测试文件夹读取图片 + dwg_edited.json 标注
  2. 从标注数据构建 classifications（与 app.py 输入格式一致）
  3. 调用 app.py 的 /calculate_associations 接口进行推理
  4. 解析接口返回的关联关系
  5. 与标注中的"从属"关系对比，计算指标

用法（需要 Flask 服务和 vLLM 都在运行）：
    cd app/
    python evaluate_associations.py

只输出 GT 统计（不调接口）：
    python evaluate_associations.py --report

调试模式（只打印要调用的请求，不发请求）：
    python evaluate_associations.py --dry-run
"""

import json
import os
import re
import sys
import time
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional

import requests
import yaml

# ==================== 硬编码配置 ====================
TEST_DATA_DIR = r"/mnt/nvme/qwen_train/app/tmp_test/_原理图训练集（第一批次）"
CONFIG_PATH = Path(__file__).parent / "config.yaml"

# ==================== 加载配置 ====================
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# ==================== 关联类型映射（与 config.yaml 一致） ====================
TYPE_MAP_CN2EN = {
    "设备-设备序号": "device-number",
    "设备-位号": "device-tag",
    "符号-位号": "symbol-tag",
    "符号-属性标签": "symbol-attr",
    # 注意：app.py 的 calculate_associations 接口不支持仪表关联
    # "仪表-位号": "instrument-tag",
}
TYPE_MAP_EN2CN = {v: k for k, v in TYPE_MAP_CN2EN.items()}
SUPPORTED_TYPES = set(TYPE_MAP_CN2EN.values())

# 额外类型（标注中有但项目未启用的）
EXTRA_TYPE_MAP = {
    "仪表-位号": "instrument-tag",
    "设备-属性标签": "device-label",
    "仪表-属性标签": "instrument-label",
    "线条-属性标签": "line-label",
    "线条-管道方向": "line-direction",
    "线条-位号": "line-tag",
    "属性标签-线条": "label-line",
    "位号-位号": "tag-tag",
    "符号-其它": "symbol-other",
    "其它-位号": "other-tag",
    "其它-属性标签": "other-label",
    "管道方向-线条": "direction-line",
    "位号-线条": "tag-line",
    "设备-设备": "device-device",
}
ALL_TYPE_MAP = {**TYPE_MAP_CN2EN, **EXTRA_TYPE_MAP}


# ==================== 坐标转换 ====================
def world_to_pixel(world_coords, world_bbox, img_size):
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


# ==================== 标注数据加载 ====================
def parse_world_bbox_from_filename(filename: str) -> Optional[Tuple[float, float, float, float]]:
    """
    从图片文件名解析世界坐标包围盒。
    文件名格式:  xxx-(-20830.52,-14712.64,20830.52,14712.64).jpg
    """
    m = re.search(r'\(([^)]+)\)\.\w+$', filename)
    if not m:
        return None
    parts = m.group(1).split(',')
    if len(parts) != 4:
        return None
    try:
        return (float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]))
    except ValueError:
        return None


def load_annotation(subfolder_path: str) -> Tuple[List[Dict], Dict[str, str], List[Dict], str, str]:
    """
    加载子文件夹中的标注文件。

    Returns:
        (nodes, node_label_map, subordination_edges, image_path, annotation_path)
    """
    json_files = [f for f in os.listdir(subfolder_path) if f.endswith("_edited.json")]
    edited_files = [f for f in json_files if not f.endswith("_transformed.json")]
    if not edited_files:
        edited_files = json_files
    if not edited_files:
        return [], {}, [], "", ""

    annotation_path = os.path.join(subfolder_path, edited_files[0])
    with open(annotation_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    nodes = data.get("nodes", [])
    edges = data.get("edges", [])

    node_label_map = {}
    for n in nodes:
        node_label_map[n["ele_id"]] = n.get("label", "")

    subordination_edges = []
    for e in edges:
        if e.get("connection") != "从属":
            continue
        src_id = e["ele_id1"]
        tgt_id = e["ele_id2"]
        src_label = node_label_map.get(src_id, "?")
        tgt_label = node_label_map.get(tgt_id, "?")
        pair_key = f"{src_label}-{tgt_label}"
        relation_type = ALL_TYPE_MAP.get(pair_key, pair_key)
        subordination_edges.append({
            "source": src_id,
            "target": tgt_id,
            "source_label": src_label,
            "target_label": tgt_label,
            "pair_key": pair_key,
            "relation_type": relation_type,
        })

    # 查找图片
    image_path = ""
    for fname in os.listdir(subfolder_path):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(subfolder_path, fname)
            break

    return nodes, node_label_map, subordination_edges, image_path, annotation_path


def build_classifications_from_nodes(
    nodes: List[Dict],
    world_bbox: Tuple[float, float, float, float],
    img_size: Tuple[int, int],
) -> List[Dict]:
    """
    从标注节点构建 API 需要的 classifications 格式。

    输入格式（标注节点）：
        {"ele_id": "...", "label": "设备", "cate": "设备", "bbox": [x1,y1,x2,y2]}

    输出格式（API 接收的 classifications）：
        {"id": "...", "category": "设备", "subclass": "...",
         "x1": ..., "y1": ..., "x2": ..., "y2": ..., "pixel_bbox": [...]}
    """
    skip_labels = {"线条", "连接点", "管道方向"}
    classifications = []

    for n in nodes:
        label = n.get("label", "")
        if label in skip_labels:
            continue

        cate = n.get("cate", "")
        bbox = n.get("bbox")
        if not bbox or not isinstance(bbox, (list, tuple)) or len(bbox) < 4 or any(v is None for v in bbox[:4]):
            continue

        # 提取 subclass
        subclass = ""
        if cate != label:
            if "-" in cate:
                subclass = cate.split("-", 1)[1] if len(cate.split("-", 1)) > 1 else ""
            else:
                subclass = cate
        if not subclass:
            subclass = label

        pixel_bbox = world_to_pixel(bbox, world_bbox, img_size)
        classifications.append({
            "id": n["ele_id"],
            "category": label,
            "subclass": subclass,
            "x1": bbox[0],
            "y1": bbox[1],
            "x2": bbox[2],
            "y2": bbox[3],
            "pixel_bbox": pixel_bbox,
        })

    return classifications


# ==================== 调用 API ====================
def call_calculate_associations(
    api_url: str,
    image_path: str,
    classifications: List[Dict],
    world_bbox: Tuple[float, float, float, float],
    timeout: int = 600,
) -> List[Dict]:
    """
    调用 app.py 的 /calculate_associations 接口。

    Args:
        api_url: 完整的 API 地址 (http://host:port/calculate_associations)
        image_path: 图片文件路径
        classifications: 元件列表（API 输入格式）
        world_bbox: (x_min, y_min, x_max, y_max)

    Returns:
        关联关系列表 [{"source": id, "target": id, "type": "device-tag"}, ...]
    """
    # 构造 classifications JSON
    # 注意：API 接收的 classifications 需要 id, x1/y1/x2/y2, category, subclass 字段
    # pixel_bbox 是可选的（API 内部会自己算）
    api_classifications = []
    for c in classifications:
        api_classifications.append({
            "id": c["id"],
            "category": c["category"],
            "subclass": c.get("subclass", c["category"]),
            "x1": c["x1"],
            "y1": c["y1"],
            "x2": c["x2"],
            "y2": c["y2"],
        })

    x_min, y_min, x_max, y_max = world_bbox

    # 构造 multipart 请求
    with open(image_path, "rb") as f:
        files = {
            "image": (os.path.basename(image_path), f, "image/jpeg"),
        }
        data = {
            "classifications": json.dumps(api_classifications, ensure_ascii=False),
            "element_lists": "[]",  # app.py 的接口不处理 element_lists，传空即可
            "x_min": str(x_min),
            "y_min": str(y_min),
            "x_max": str(x_max),
            "y_max": str(y_max),
        }

        resp = requests.post(api_url, files=files, data=data, timeout=timeout)

    if resp.status_code != 200:
        raise RuntimeError(f"API 返回错误状态码 {resp.status_code}: {resp.text[:500]}")

    result = resp.json()
    if not result.get("success"):
        raise RuntimeError(f"API 返回失败: {result.get('error', '未知错误')}")

    associations = result.get("associations", [])
    # 标准化：确保每个关联有 source/target/type
    normalized = []
    for a in associations:
        normalized.append({
            "source": a.get("source", ""),
            "target": a.get("target", ""),
            "type": a.get("type", "unknown"),
        })

    return normalized


# ==================== 评估指标计算 ====================
def compute_confusion_matrix(gt_edges, pred_edges):
    """计算混淆矩阵（按 source+target+type 三元组匹配）"""
    gt_set = set()
    for e in gt_edges:
        gt_set.add((e["source"], e["target"], e["relation_type"]))

    pred_set = set()
    for e in pred_edges:
        pred_set.add((e["source"], e["target"], e["type"]))

    tp = len(gt_set & pred_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)

    gt_by_type = defaultdict(set)
    for e in gt_edges:
        gt_by_type[e["relation_type"]].add((e["source"], e["target"]))

    pred_by_type = defaultdict(set)
    for e in pred_edges:
        pred_by_type[e["type"]].add((e["source"], e["target"]))

    type_metrics = {}
    all_types = set(gt_by_type.keys()) | set(pred_by_type.keys())
    for t in sorted(all_types):
        gt_pairs = gt_by_type.get(t, set())
        pred_pairs = pred_by_type.get(t, set())
        t_tp = len(gt_pairs & pred_pairs)
        t_fp = len(pred_pairs - gt_pairs)
        t_fn = len(gt_pairs - pred_pairs)
        precision = t_tp / (t_tp + t_fp) if (t_tp + t_fp) > 0 else 0
        recall = t_tp / (t_tp + t_fn) if (t_tp + t_fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        type_metrics[t] = {
            "tp": t_tp, "fp": t_fp, "fn": t_fn,
            "gt_count": len(gt_pairs), "pred_count": len(pred_pairs),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "total": {
            "tp": tp, "fp": fp, "fn": fn,
            "gt_count": len(gt_set), "pred_count": len(pred_set),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        },
        "by_type": type_metrics,
    }


def get_short_name(folder_name):
    """提取简短文件夹名"""
    parts = folder_name.split("_", 3)
    if len(parts) >= 4:
        return "_".join(parts[3:])
    return folder_name


# ==================== 主流程 ====================
def main():
    test_dir = TEST_DATA_DIR
    if not os.path.exists(test_dir):
        print(f"❌ 测试数据目录不存在: {test_dir}")
        sys.exit(1)

    dry_run = "--dry-run" in sys.argv
    report_only = "--report" in sys.argv

    subfolders = sorted([
        d for d in os.listdir(test_dir)
        if os.path.isdir(os.path.join(test_dir, d))
    ])

    print(f"{'='*70}")
    print(f"  关联关系识别端到端评估（调用 API）")
    print(f"{'='*70}")
    print(f"测试数据: {test_dir}")
    print(f"子文件夹: {len(subfolders)} 个")

    if dry_run:
        print(f"模式: DRY RUN (不发请求)")
    elif report_only:
        print(f"模式: 仅输出 GT 统计")

    print()

    # ========== 加载所有标注 ==========
    folder_data = {}
    all_gt_edges = []

    for folder in subfolders:
        folder_path = os.path.join(test_dir, folder)
        nodes, label_map, gt_edges, image_path, ann_path = load_annotation(folder_path)
        folder_data[folder] = {
            "nodes": nodes,
            "label_map": label_map,
            "gt_edges": gt_edges,
            "image_path": image_path,
            "ann_path": ann_path,
        }
        all_gt_edges.extend(gt_edges)

    # ========== GT 统计 ==========
    overall_type_counter = Counter(e["relation_type"] for e in all_gt_edges)
    print(f"{'='*70}")
    print(f"  Ground Truth 统计")
    print(f"{'='*70}")
    print(f"总从属关系数: {len(all_gt_edges)}")
    print()
    print("各关联类型分布:")
    for t, cnt in sorted(overall_type_counter.items(), key=lambda x: -x[1]):
        if t in SUPPORTED_TYPES:
            mark = "✓ API支持"
        elif t == "instrument-tag":
            mark = "✗ API不支持（仪表过滤）"
        else:
            mark = "  (非项目类型)"
        print(f"  [{mark}] {t:25s}: {cnt:4d} ({cnt/len(all_gt_edges)*100:5.1f}%)")

    print()
    print("API 支持的关联类型:")
    for t in sorted(SUPPORTED_TYPES):
        cnt = overall_type_counter.get(t, 0)
        ch_name = TYPE_MAP_EN2CN.get(t, t)
        print(f"  {t:20s} ({ch_name}): {cnt} 条 GT")
    print()

    if report_only:
        report = {}
        for folder in subfolders:
            fd = folder_data[folder]
            tc = Counter(e["relation_type"] for e in fd["gt_edges"])
            lc = Counter(fd["label_map"].values())
            report[folder] = {
                "short_name": get_short_name(folder),
                "nodes": len(fd["nodes"]),
                "gt_edges": len(fd["gt_edges"]),
                "types": dict(tc),
                "labels": dict(lc),
            }
        report_path = "gt_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"GT 报告已保存: {report_path}")
        return

    # ========== 初始化 API 客户端 ==========
    flask_config = config["flask"]
    api_host = flask_config['host']
    if api_host == "0.0.0.0":
        api_host = "127.0.0.1"
    api_base_url = f"http://{api_host}:{flask_config['port']}"
    api_url = f"{api_base_url}/calculate_associations"

    if not dry_run:
        try:
            # 检查 API 健康状态
            health_url = f"{api_base_url}/health"
            resp = requests.get(health_url, timeout=10)
            health = resp.json()
            print(f"✅ API 服务连接成功")
            print(f"   地址: {api_url}")
            print(f"   状态: {health.get('status', 'OK')}")
            print()
        except Exception as e:
            print(f"❌ API 服务连接失败: {e}")
            print(f"请先启动 Flask 服务: python app.py")
            sys.exit(1)
    else:
        api_url = ""

    # ========== 逐文件夹调用 API + 评估 ==========
    overall_results = {}
    all_predictions = {}

    for folder in subfolders:
        fd = folder_data[folder]
        short = get_short_name(folder)

        print(f"\n{'='*70}")
        print(f"  📁 {short}")
        print(f"{'='*70}")

        if not fd["gt_edges"]:
            print(f"  跳过: 无 GT 数据")
            continue

        image_path = fd["image_path"]
        if not image_path or not os.path.exists(image_path):
            print(f"  ❌ 跳过: 找不到图片")
            continue

        # 获取图片尺寸
        try:
            from PIL import Image as PILImage
            img = PILImage.open(image_path)
            img_size = img.size
            img.close()
            print(f"  图片: {os.path.basename(image_path)} ({img_size[0]}x{img_size[1]})")
        except Exception as e:
            print(f"  ❌ 图片加载失败: {e}")
            continue

        # 解析世界坐标
        world_bbox = parse_world_bbox_from_filename(os.path.basename(image_path))
        if not world_bbox:
            print(f"  ❌ 无法从文件名解析世界坐标")
            continue
        print(f"  世界坐标: ({world_bbox[0]:.1f}, {world_bbox[1]:.1f}, {world_bbox[2]:.1f}, {world_bbox[3]:.1f})")

        # 构建 classifications
        classifications = build_classifications_from_nodes(fd["nodes"], world_bbox, img_size)
        cat_counts = Counter(c['category'] for c in classifications)
        print(f"  元件: {len(classifications)} 个")
        cat_str = ", ".join(f"{k}={v}" for k, v in sorted(cat_counts.items(), key=lambda x: -x[1]))
        print(f"  类别: {cat_str}")

        # 调用 API
        if dry_run:
            print(f"  模拟调用 API: {api_url}")
            print(f"    classifications: {len(classifications)} 个元件")
            print(f"    world_bbox: {world_bbox}")
            all_predictions[folder] = []
            continue

        t_start = time.time()
        try:
            predictions = call_calculate_associations(
                api_url, image_path, classifications, world_bbox,
                timeout=1800,  # 大图可能处理很久
            )
            elapsed = time.time() - t_start
        except Exception as e:
            print(f"  ❌ API 调用失败: {e}")
            continue

        all_predictions[folder] = predictions

        # 统计预测类型
        pred_types = Counter(a['type'] for a in predictions)
        pred_str = ", ".join(f"{k}={v}" for k, v in sorted(pred_types.items()))
        print(f"  预测关联: {len(predictions)} 条 [{pred_str}]")
        print(f"  耗时: {elapsed:.1f}s")

        # 计算指标
        metrics = compute_confusion_matrix(fd["gt_edges"], predictions)
        overall_results[folder] = metrics

        t = metrics["total"]
        print(f"  指标: GT={t['gt_count']} Pred={t['pred_count']} "
              f"P={t['precision']:.4f} R={t['recall']:.4f} F1={t['f1']:.4f}")

        for tname in sorted(SUPPORTED_TYPES):
            if tname not in metrics["by_type"]:
                continue
            tm = metrics["by_type"][tname]
            print(f"    [{tname:20s}] GT={tm['gt_count']:3d} Pred={tm['pred_count']:3d}  "
                  f"P={tm['precision']:.4f} R={tm['recall']:.4f} F1={tm['f1']:.4f}")

    # ========== 汇总指标 ==========
    if not dry_run and overall_results:
        print(f"\n\n{'='*70}")
        print(f"  汇总指标")
        print(f"{'='*70}\n")

        type_agg = {}
        for metrics in overall_results.values():
            for tname, tm in metrics["by_type"].items():
                if tname not in type_agg:
                    type_agg[tname] = {"tp": 0, "fp": 0, "fn": 0, "gt_count": 0, "pred_count": 0}
                type_agg[tname]["tp"] += tm["tp"]
                type_agg[tname]["fp"] += tm["fp"]
                type_agg[tname]["fn"] += tm["fn"]

        total_tp = sum(agg["tp"] for agg in type_agg.values())
        total_fp = sum(agg["fp"] for agg in type_agg.values())
        total_fn = sum(agg["fn"] for agg in type_agg.values())

        print(f"  API 支持的 4 种类型:")
        print(f"  {'类型':25s} {'GT':>5s} {'Pred':>5s} {'TP':>4s} {'FP':>4s} {'FN':>4s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s}")
        print(f"  {'-'*25} {'-'*5} {'-'*5} {'-'*4} {'-'*4} {'-'*4} {'-'*10} {'-'*10} {'-'*10}")

        for tname in sorted(SUPPORTED_TYPES):
            agg = type_agg.get(tname)
            if agg is None:
                cnt = overall_type_counter.get(tname, 0)
                print(f"  {tname:25s}  GT 有 {cnt} 条但 API 未预测任何该类型")
                continue
            p = agg["tp"] / (agg["tp"] + agg["fp"]) if (agg["tp"] + agg["fp"]) > 0 else 0
            r = agg["tp"] / (agg["tp"] + agg["fn"]) if (agg["tp"] + agg["fn"]) > 0 else 0
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0
            print(f"  {tname:25s} {agg['tp']+agg['fn']:5d} {agg['tp']+agg['fp']:5d} {agg['tp']:4d} {agg['fp']:4d} {agg['fn']:4d} {p:10.4f} {r:10.4f} {f:10.4f}")

        # 其他类型（包括 instrument-tag）
        other_types = {t: agg for t, agg in type_agg.items() if t not in SUPPORTED_TYPES}
        if other_types:
            print()
            print(f"  其他类型（标注中有但 API 不支持的）:")
            for tname in sorted(other_types.keys()):
                agg = other_types[tname]
                p = agg["tp"] / (agg["tp"] + agg["fp"]) if (agg["tp"] + agg["fp"]) > 0 else 0
                r = agg["tp"] / (agg["tp"] + agg["fn"]) if (agg["tp"] + agg["fn"]) > 0 else 0
                f = 2 * p * r / (p + r) if (p + r) > 0 else 0
                note = " [API 不支持-仪表过滤]" if tname == "instrument-tag" else ""
                print(f"    {tname:25s} {agg['tp']+agg['fn']:5d} {agg['tp']+agg['fp']:5d} {agg['tp']:4d} {agg['fp']:4d} {agg['fn']:4d} {p:10.4f} {r:10.4f} {f:10.4f}{note}")

        # 总结（只算 API 支持的 4 种）
        supported_tp = sum(type_agg[t]["tp"] for t in SUPPORTED_TYPES if t in type_agg)
        supported_fp = sum(type_agg[t]["fp"] for t in SUPPORTED_TYPES if t in type_agg)
        supported_fn = sum(type_agg[t]["fn"] for t in SUPPORTED_TYPES if t in type_agg)
        supported_precision = supported_tp / (supported_tp + supported_fp) if (supported_tp + supported_fp) > 0 else 0
        supported_recall = supported_tp / (supported_tp + supported_fn) if (supported_tp + supported_fn) > 0 else 0
        supported_f1 = 2 * supported_precision * supported_recall / (supported_precision + supported_recall) if (supported_precision + supported_recall) > 0 else 0

        print(f"  {'-'*25} {'-'*5} {'-'*5} {'-'*4} {'-'*4} {'-'*4} {'-'*10} {'-'*10} {'-'*10}")
        print(f"  {'API 支持合计':25s} {supported_tp+supported_fn:5d} {supported_tp+supported_fp:5d} {supported_tp:4d} {supported_fp:4d} {supported_fn:4d} {supported_precision:10.4f} {supported_recall:10.4f} {supported_f1:10.4f}")

        print()
        print(f"  (instrument-tag 的 222 条 GT 不在上述统计中，因为 API 不识别仪表关联)")

        # ========== 保存结果 ==========
        output = {
            "config": {
                "api_url": api_url,
                "test_data_dir": test_dir,
                "api_supported_types": list(SUPPORTED_TYPES),
            },
            "summary": {
                "total_gt_supported": supported_tp + supported_fn,
                "total_pred_supported": supported_tp + supported_fp,
                "total_tp": supported_tp,
                "total_fp": supported_fp,
                "total_fn": supported_fn,
                "overall_precision": supported_precision,
                "overall_recall": supported_recall,
                "overall_f1": supported_f1,
            },
            "by_subfolder": {},
            "by_type": {},
        }

        for folder, metrics in overall_results.items():
            t = metrics["total"]
            output["by_subfolder"][folder] = {
                "short_name": get_short_name(folder),
                "nodes": len(folder_data[folder]["nodes"]),
                "gt_count": t["gt_count"],
                "pred_count": t["pred_count"],
                "tp": t["tp"], "fp": t["fp"], "fn": t["fn"],
                "precision": t["precision"],
                "recall": t["recall"],
                "f1": t["f1"],
            }

        for tname, agg in type_agg.items():
            p = agg["tp"] / (agg["tp"] + agg["fp"]) if (agg["tp"] + agg["fp"]) > 0 else 0
            r = agg["tp"] / (agg["tp"] + agg["fn"]) if (agg["tp"] + agg["fn"]) > 0 else 0
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0
            ch_name = TYPE_MAP_EN2CN.get(tname, ALL_TYPE_MAP.get(tname, tname))
            output["by_type"][tname] = {
                "chinese_name": ch_name,
                "gt_count": agg["tp"] + agg["fn"],
                "pred_count": agg["tp"] + agg["fp"],
                "tp": agg["tp"], "fp": agg["fp"], "fn": agg["fn"],
                "precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4),
            }

        output_path = "evaluation_results.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"评估结果已保存: {output_path}")

    # ========== 保存预测结果 ==========
    if all_predictions:
        pred_path = "predictions.json"
        with open(pred_path, "w", encoding="utf-8") as f:
            json.dump(all_predictions, f, ensure_ascii=False, indent=2)
        print(f"预测结果已保存: {pred_path}")
        if dry_run:
            # dry_run 模式删掉空文件
            os.remove(pred_path)

    # ========== 结论 ==========
    if not dry_run and overall_results:
        print(f"\n{'='*70}")
        print(f"  结论")
        print(f"{'='*70}")
        print(f"  评估方式: 调用 /calculate_associations API")
        print(f"  API 支持 4 种类型（不含 instrument-tag）")
        print(f"  GT 中 API 支持的类型数: {supported_tp + supported_fn}")
        print(f"  总预测关系数: {supported_tp + supported_fp}")
        print(f"  总 Precision: {supported_precision:.4f} ({supported_precision*100:.2f}%)")
        print(f"  总 Recall:    {supported_recall:.4f} ({supported_recall*100:.2f}%)")
        print(f"  总 F1:        {supported_f1:.4f} ({supported_f1*100:.2f}%)")
        print()


if __name__ == "__main__":
    main()
