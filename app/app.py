"""
Flask服务 - 处理数据并调用vLLM进行推理
Flask端口: 5004
vLLM端口: 5003
"""
from flask import Flask, request, jsonify
from pathlib import Path
import yaml
from PIL import Image
from io import BytesIO
from openai import OpenAI

from utils import (
    world_to_pixel,
    crop_image_for_element,
    crop_image_for_element_device,
    draw_boxes_on_image,
    resize_image,
    build_prompt_for_inference,
    parse_model_output,
    convert_to_association_format,
    select_candidates,
)


def load_config():
    config_path = Path(__file__).parent / 'config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


config = load_config()
app = Flask(__name__)

# 初始化vLLM客户端
vllm_base_url = config['model']['vllm_base_url']
vllm_model_name = config['model']['vllm_model_name']
vllm_client = OpenAI(api_key="EMPTY", base_url=vllm_base_url)

print(f"✅ Flask服务已初始化")
print(f"   - vLLM地址: {vllm_base_url}")
print(f"   - 模型名称: {vllm_model_name}")


@app.route('/health', methods=['GET'])
def health():
    """健康检查"""
    try:
        # 检查vLLM服务
        models = vllm_client.models.list()
        model_ids = [m.id for m in models.data]
        
        return jsonify({
            'success': True,
            'service': 'Flask + vLLM',
            'flask_port': config['flask']['port'],
            'vllm_url': vllm_base_url,
            'vllm_models': model_ids,
            'status': 'ready'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'status': 'vLLM service unavailable'
        }), 503


@app.route('/calculate_associations', methods=['POST'])
def calculate_associations():
    """
    计算关联关系接口（与旧接口兼容）
    接收图片和元件数据，返回关联关系
    """
    try:
        print(f"收到请求 - Content-Type: {request.content_type}")
        print(f"Files keys: {list(request.files.keys())}")
        print(f"Form keys: {list(request.form.keys())}")
        
        # 1. 解析请求参数
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': '缺少图片'}), 400
        
        image_file = request.files['image']
        print(f"图片信息 - 文件名: '{image_file.filename}', Content-Type: {image_file.content_type}")
        
        classifications = request.form.get('classifications')
        
        if not classifications:
            return jsonify({'success': False, 'error': '缺少元件数据'}), 400
        
        # 解析元件数据
        import json
        try:
            classifications = json.loads(classifications)
        except json.JSONDecodeError as e:
            return jsonify({'success': False, 'error': f'元件数据JSON解析失败: {str(e)}'}), 400
        
        # 解析坐标范围
        try:
            x_min = float(request.form.get('x_min', 0))
            y_min = float(request.form.get('y_min', 0))
            x_max = float(request.form.get('x_max', 0))
            y_max = float(request.form.get('y_max', 0))
        except ValueError as e:
            return jsonify({'success': False, 'error': f'坐标参数错误: {str(e)}'}), 400
        
        # 2. 加载图片
        try:
            # 重置文件指针到开头（防止stream已被读取）
            image_file.seek(0)
            image = Image.open(image_file).convert('RGB')
            print(f"✅ 图片加载成功 - 文件名: {image_file.filename}, 尺寸: {image.size}")
        except Exception as e:
            print(f"❌ 图片加载失败: {str(e)}")
            return jsonify({'success': False, 'error': f'图片加载失败: {str(e)}'}), 400
        img_size = image.size
        world_bbox = (x_min, y_min, x_max, y_max)
        
        # 3. 转换坐标
        print(f"\n{'='*60}")
        print(f"📊 数据统计")
        print(f"{'='*60}")
        print(f"元件总数: {len(classifications)}")
        
        # 统计元件类别
        category_counts = {}
        for elem in classifications:
            cat = elem.get('category', '未知')
            category_counts[cat] = category_counts.get(cat, 0) + 1
        print(f"元件类别分布:")
        for cat, count in sorted(category_counts.items()):
            print(f"  - {cat}: {count}个")
        
        for elem in classifications:
            pixel_bbox = world_to_pixel(
                [elem['x1'], elem['y1'], elem['x2'], elem['y2']],
                world_bbox,
                img_size
            )
            elem['pixel_bbox'] = pixel_bbox
            
            # 保存原始类别名称（用于返回）
            if 'category' not in elem or not elem['category']:
                elem['category'] = elem.get('subclass', '其它')
            elem['original_category'] = elem['category']  # 保存原始类别
        
        # 4. 筛选主体元件（设备 + 小类名称带"阀"字的符号）
        print(f"\n{'='*60}")
        print(f"🔍 筛选主体元件")
        print(f"{'='*60}")

        main_elements = []
        for elem in classifications:
            if elem['category'] == '设备':
                main_elements.append(elem)
            elif elem['category'] == '符号':
                subclass = elem.get('subclass', '').strip()
                if '阀' in subclass:
                    main_elements.append(elem)

        print(f"原始元件总数: {len(classifications)}个")
        print(f"主体元件（设备/带阀符号）: {len(main_elements)}个")
        
        if not main_elements:
            print("⚠️  未找到主体元件，返回空结果")
            return jsonify({
                'success': True,
                'associations': [],
                'message': '未找到主体元件'
            })
        
        # 5. 处理每个主体元件
        print(f"\n{'='*60}")
        print(f"🔍 开始处理主体元件")
        print(f"{'='*60}")
        
        all_associations = []
        category_map = config['association']['category_map']
        max_candidates = config['image']['max_candidates']
        crop_size = config['image']['crop_size']
        default_conf = config['association']['default_confidence']
        
        processed_count = 0  # 成功处理的数量
        skipped_no_candidates = 0  # 因无候选元件跳过的数量
        failed_count = 0  # 推理失败的数量
        cropped_images_count = 0  # 裁剪的图片数量
        
        for idx, main_elem in enumerate(main_elements, 1):
            elem_id = main_elem.get('id', 'unknown')
            elem_cat = main_elem.get('category', 'unknown')
            
            # 选择候选元件，并针对设备做特殊处理
            if elem_cat == '设备':
                dx1, dy1, dx2, dy2 = main_elem['pixel_bbox']
                # 找出完全在设备包围框内的位号和设备序号
                inner_elems = [
                    e for e in classifications
                    if e.get('category') in ('位号', '设备序号')
                    and e['pixel_bbox'][0] >= dx1 and e['pixel_bbox'][1] >= dy1
                    and e['pixel_bbox'][2] <= dx2 and e['pixel_bbox'][3] <= dy2
                ]
                use_device_crop = len(inner_elems) > 0
                if use_device_crop:
                    # 内部元件优先，再补充距离最近的其他候选
                    inner_ids = {e['id'] for e in inner_elems}
                    extra = select_candidates(main_elem, classifications, max_candidates)
                    extra = [e for e in extra if e['id'] not in inner_ids]
                    candidates = inner_elems + extra
                    candidates = candidates[:max_candidates]
                else:
                    candidates = select_candidates(main_elem, classifications, max_candidates)
            else:
                use_device_crop = False
                candidates = select_candidates(main_elem, classifications, max_candidates)

            if not candidates:
                skipped_no_candidates += 1
                print(f"[{idx}/{len(main_elements)}] 跳过 {elem_id} ({elem_cat}) - 无候选元件")
                continue

            # 调试：打印候选元件的类型
            candidate_categories = [c.get('category', 'unknown') for c in candidates]
            print(f"[{idx}/{len(main_elements)}] 处理 {elem_id} ({elem_cat}) - 候选元件: {len(candidates)}个")
            print(f"   候选元件类型: {candidate_categories}")

            # 分配简单ID
            main_elem['simple_id'] = 1
            for i, cand in enumerate(candidates):
                cand['simple_id'] = i + 2

            # 裁剪图片
            try:
                if use_device_crop:
                    # 设备内有位号/设备序号：用设备扩展裁剪
                    cropped_img, updated_main, updated_candidates = crop_image_for_element_device(
                        image, main_elem, candidates, main_elem['pixel_bbox']
                    )
                    print(f"   📦 设备扩展裁剪（内部有{len(inner_elems)}个位号/设备序号）")
                else:
                    # 设备内无位号/设备序号，或符号主体：中心固定裁剪
                    cropped_img, updated_main, updated_candidates = crop_image_for_element(
                        image, main_elem, candidates, crop_size
                    )
                    if elem_cat == '设备':
                        print(f"   🔲 设备中心裁剪（内部无位号/设备序号）")
                cropped_images_count += 1
            except Exception as e:
                print(f"   ❌ 图片裁剪失败: {e}")
                failed_count += 1
                continue
            
            # 绘制边框
            img_with_boxes = draw_boxes_on_image(cropped_img, updated_main, updated_candidates)
            img_resized = resize_image(img_with_boxes, crop_size)
            
            # 转换为base64
            import base64
            buffered = BytesIO()
            img_resized.save(buffered, format="JPEG")
            base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            # 构建提示词
            prompt = build_prompt_for_inference(updated_main, updated_candidates, category_map)
            
            # 调用vLLM推理
            try:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                            {"type": "text", "text": prompt}
                        ]
                    }
                ]
                
                # 使用与evaluate.py一致的参数
                response = vllm_client.chat.completions.create(
                    model=vllm_model_name,
                    messages=messages,
                    max_tokens=16,      # 与evaluate.py一致
                    temperature=0.0,    # 贪婪解码
                    top_p=1.0,
                )
                
                output_text = response.choices[0].message.content
                target_ids = parse_model_output(output_text)
                
                # 转换为关联格式（包含过滤逻辑，使用几何置信度）
                associations = convert_to_association_format(
                    main_elem, target_ids, candidates, default_conf, 
                    use_geometry_confidence=True  # 启用基于几何关系的置信度
                )
                
                # 显示过滤信息
                filtered_count = len(target_ids) - len(associations)
                if filtered_count > 0:
                    print(f"   ⚠️  过滤不合理关联: {filtered_count}个")
                
                all_associations.extend(associations)
                
                processed_count += 1
                print(f"   ✅ 推理成功 - 模型输出: '{output_text.strip()}' -> 有效关联: {len(associations)}个")
                
            except Exception as e:
                failed_count += 1
                print(f"   ❌ 推理失败: {e}")
                import traceback
                traceback.print_exc()
                continue

        # 6. 返回结果
        print(f"\n{'='*60}")
        print(f"📈 处理结果统计")
        print(f"{'='*60}")
        print(f"主体元件总数: {len(main_elements)}个")
        print(f"成功处理: {processed_count}个")
        print(f"裁剪图片数: {cropped_images_count}个")
        print(f"跳过（无候选元件）: {skipped_no_candidates}个")
        print(f"推理失败: {failed_count}个")
        print(f"生成关联关系: {len(all_associations)}个")
        print(f"{'='*60}\n")
        
        return jsonify({
            'success': True,
            'associations': all_associations,
            'stats': {
                'total_elements': len(classifications),
                'main_elements': len(main_elements),
                'processed': processed_count,
                'cropped_images': cropped_images_count,
                'skipped': skipped_no_candidates,
                'failed': failed_count,
                'associations_count': len(all_associations)
            }
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    flask_config = config['flask']
    host = flask_config['host']
    port = flask_config['port']
    debug = flask_config['debug']
    
    print(f"\n{'='*60}")
    print(f"🚀 启动Flask服务")
    print(f"{'='*60}")
    print(f"服务地址: http://{host}:{port}")
    print(f"健康检查: http://{host}:{port}/health")
    print(f"推理接口: http://{host}:{port}/calculate_associations")
    print(f"{'='*60}\n")
    
    app.run(host=host, port=port, debug=debug)

