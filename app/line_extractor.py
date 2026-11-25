"""
线条提取封装模块
复用 build_graph.py 的线条标准化、提取、过滤逻辑。
后续升级 build_graph.py 后，此模块自动获得更新。
"""
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set

# 添加 qingtian-ai 路径
_GRAPH_DIR = Path(__file__).parent.parent.parent / 'qingtian-ai' / 'yolo'
_GRAPH_PKG = _GRAPH_DIR / 'graph'
if str(_GRAPH_DIR) not in sys.path:
    sys.path.insert(0, str(_GRAPH_DIR))

from graph.build_graph import (
    normalize_line_entity,
    extract_lines_in_region,
    split_polylines_to_segments,
    filter_internal_lines,
    CONNECTABLE_CATEGORIES,
)


def extract_lines(
    entities: List[Dict],
    region: Optional[Tuple[float, float, float, float]] = None,
    components: Optional[List[Dict]] = None,
    line_types: Optional[Set[str]] = None,
) -> List[Dict]:
    """
    从 DWG 图元列表中提取、标准化并过滤线条。

    Args:
        entities: 原始 DWG 图元列表（ODA 格式）
        region: 有效区域 (x_min, y_min, x_max, y_max)，None 则不过滤区域
        components: 检测到的元件列表（用于过滤元件内部线条）
        line_types: 允许的线条类型集合，默认 {"line", "polyline", "arc"}

    Returns:
        标准化后的线段列表，每条格式:
        {"type": "line", "points": [[x1,y1], [x2,y2]], "id": "原始ID", "idx": 索引}
    """
    if line_types is None:
        line_types = {"line", "polyline", "arc"}

    if region:
        lines, _ = extract_lines_in_region(entities, region, line_types=line_types)
    else:
        # 不过滤区域，只做标准化
        lines = []
        for entity in entities:
            if entity.get("type", "") not in line_types:
                continue
            normalized = normalize_line_entity(entity)
            if normalized:
                normalized["idx"] = len(lines)
                lines.append(normalized)

    # Polyline 拆分为独立线段
    lines = split_polylines_to_segments(lines)

    # 过滤元件内部的线条（如位号基线等）
    if components:
        lines, _ = filter_internal_lines(lines, components)

    return lines


def entities_in_region(
    entities: List[Dict],
    region: Tuple[float, float, float, float],
    components: Optional[List[Dict]] = None,
) -> Tuple[List[Dict], List[Dict]]:
    """
    获取指定区域内的线条，并返回过滤前后的结果。

    Returns:
        (filtered_lines, raw_lines_in_region)
    """
    x_min, y_min, x_max, y_max = region
    raw_lines = []
    for entity in entities:
        etype = entity.get("type", "")
        if etype not in ("line", "polyline", "arc"):
            continue
        normalized = normalize_line_entity(entity)
        if not normalized:
            continue
        in_region = any(
            x_min <= pt[0] <= x_max and y_min <= pt[1] <= y_max
            for pt in normalized["points"]
        )
        if in_region:
            normalized["idx"] = len(raw_lines)
            raw_lines.append(normalized)

    raw_lines = split_polylines_to_segments(raw_lines)
    filtered = raw_lines
    if components:
        filtered, _ = filter_internal_lines(raw_lines, components)

    return filtered, raw_lines
