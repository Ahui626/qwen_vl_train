"""
数据集模块
简化的训练数据格式：每条关联关系一个样本
"""
import json
import random
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


# 训练图片最大尺寸
MAX_IMAGE_SIZE = 1024


def resize_image(image: Image.Image, max_size: int = MAX_IMAGE_SIZE) -> Image.Image:
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


def build_prompt(main_element: dict, candidate_elements: list) -> str:
    """
    构建工程CAD图纸关联识别提示词
    """
    # 类别中英文映射
    category_map = {
        '符号': 'symbol',
        '设备': 'device',
        '仪表': 'instrument',
        '位号': 'tag',
        '设备序号': 'device_number',
        '属性标签': 'label'
    }
    
    # 构建元件信息
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
    
    # 转为JSON字符串
    import json
    elements_json = json.dumps(elements_info, ensure_ascii=False, indent=2)
    
    prompt = f"""你是工程CAD图纸中的【元件关联识别助手】。

任务：
判断【红色框位号】与【黄色框元件】是否存在工程上的直接关联关系，
并从黄色框中选出 1～2 个与红色框位号最相关的元件。

关联关系的判定标准（满足其一即可）：
1. 黄色框明显被红色框位号标注或指向
2. 黄色框在工程语义上属于红色框位号所标识的对象
3. 空间位置上明显围绕、贴近或对齐红色框，且具有工程含义
4. 工程常识上符合"tag-device / tag-symbol / tag-label / tag-device_number"的组合

不属于关联的情况（必须排除）：
- 仅空间接近但无工程语义
- 黄色框元件属于其他位号标识
- 无法确定工程指向关系
- 位号关联了位号

输入说明：
- 红色框：查询主体位号（ID1）
- 黄色框：候选关联元件（设备、符号、属性标签、设备序号）
- 框左上角绿色数字为元件ID
- bbox坐标：以图中的左上角为原点的坐标系

输入数据格式：
{elements_json}

输出要求（严格遵守）：
- 如果存在关联：只输出黄色框的ID数字，按相关性从高到低排序，多个用英文逗号分隔
- 如果关联**2个同类别元件**只选最相关1个
- 禁止输出任何其他文字、解释、标点或空格
"""
    
    return prompt


def build_answer(answer: dict) -> str:
    """构建答案 - 支持1-2个关联ID，无关联时返回"NONE" """
    # answer 可能是单个ID或ID列表
    if isinstance(answer.get('target_ids'), list):
        # 多个关联
        if len(answer['target_ids']) == 0:
            return "NONE"  # 没有关联关系（与提示词一致）
        return ", ".join(str(tid) for tid in answer['target_ids'])
    else:
        # 单个关联（兼容旧格式）
        target_id = answer.get('target_id', answer.get('target_ids', ''))
        if not target_id:
            return "NONE"
        return str(target_id)


class RelationDataset(Dataset):
    """关联关系数据集（每条关联一个样本）"""
    
    def __init__(self, data_dir: str, processor, split: str = 'train', train_ratio: float = 0.9):
        self.data_dir = Path(data_dir)
        self.processor = processor
        self.samples = []
        self.sample_metadata = []  # 保存原始answer等元数据
        
        # 扫描所有JSON文件（支持 sample_*.json, valve_*.json, device_*.json, tag_*.json）
        all_json_files = []
        for subfolder in self.data_dir.iterdir():
            if subfolder.is_dir():
                # 支持多种文件命名格式
                for pattern in ['sample_*.json', 'valve_*.json', 'device_*.json', 'tag_*.json']:
                    for json_file in subfolder.glob(pattern):
                        all_json_files.append(str(json_file))
        
        all_json_files.sort()
        print(f"Found {len(all_json_files)} samples total")
        
        # 划分数据集
        random.seed(42)
        indices = list(range(len(all_json_files)))
        random.shuffle(indices)
        split_idx = int(len(indices) * train_ratio)
        
        if split == 'train':
            selected_indices = indices[:split_idx]
        else:
            selected_indices = indices[split_idx:]
        
        self.samples = [all_json_files[i] for i in selected_indices]
        
        # 预加载元数据（用于评估时获取真实标签）
        for json_file in self.samples:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.sample_metadata.append({
                    'answer': data.get('answer', [])
                })
        
        print(f"Loaded {len(self.samples)} samples for {split}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        json_path = Path(self.samples[idx])
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 加载图片
        image_path = json_path.parent / data['image']
        image = Image.open(image_path).convert('RGB')
        image = resize_image(image)
        
        # 构建提示词和答案
        question = build_prompt(data['main_element'], data['candidate_elements'])
        answer = build_answer(data['answer'])
        
        # 构建对话消息
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer}],
            },
        ]
        
        # 处理输入
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        )
        
        # 移除batch维度
        result = {}
        for k, v in inputs.items():
            if k == 'image_grid_thw':
                result[k] = v.squeeze(0) if v.dim() > 2 else v
            else:
                result[k] = v.squeeze(0)
        return result


def create_datasets(data_dir: str, processor, train_ratio: float = 0.9):
    """创建训练集和测试集"""
    train_dataset = RelationDataset(data_dir, processor, split='train', train_ratio=train_ratio)
    test_dataset = RelationDataset(data_dir, processor, split='test', train_ratio=train_ratio)
    return train_dataset, test_dataset

