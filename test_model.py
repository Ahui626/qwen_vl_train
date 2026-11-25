"""
测试基础模型
输入图片和提示词，查看模型输出
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES', '2')

import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor


def test_model(image_path: str, prompt: str, model_path: str = "/home/ubuntu/model/Qwen2.5-VL-3B-Instruct"):
    """
    测试模型
    
    Args:
        image_path: 图片路径
        prompt: 提示词
        model_path: 模型路径（默认使用基础模型）
    """
    print("="*80)
    print("模型测试")
    print("="*80)
    print(f"模型路径: {model_path}")
    print(f"图片路径: {image_path}")
    print(f"提示词: {prompt}")
    print()
    
    # 加载图片
    print("加载图片...")
    try:
        image = Image.open(image_path).convert('RGB')
        print(f"图片尺寸: {image.size}")
    except Exception as e:
        print(f"错误: 无法加载图片 - {e}")
        return
    
    # 加载模型和处理器
    print("\n加载模型...")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print("模型加载完成")
    
    # 构建输入
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        },
    ]
    
    # 处理输入
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    
    # 生成回答
    print("\n生成回答中...")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
        )
    
    # 解码输出
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(inputs.input_ids, output_ids)
    ]
    response = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]
    
    # 显示结果
    print("\n" + "="*80)
    print("模型回答:")
    print("="*80)
    print(response)
    print("="*80)
    
    return response


if __name__ == '__main__':
    # 图片路径（请修改为实际路径）
    image_path = "/home/ubuntu/qwen_train/processed_data/10个项目_项目1_系统图_精处理系统图 - 酸碱再生/sample_0000.jpg"
    
    # 提示词
    prompt = """识别图中所有彩色方框，提取每个框左上角的绿色数字ID和框的颜色，输出一个数字和对应颜色的列表。

要求：
1. 只关注方框和左上角的绿色数字
2. 忽略框内其他内容
3. 输出一个数字和对应颜色的列表
"""
    
    # 检查图片路径
    if not image_path:
        print("错误: 请在代码中设置 image_path 变量")
        print("例如: image_path = 'processed_data/xxx/crop_0000.jpg'")
        exit(1)
    
    # 运行测试
    test_model(image_path, prompt)
