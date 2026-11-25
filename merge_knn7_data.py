"""
合并knn7和非knn7的预处理数据
将knn7数据复制到非knn7数据文件夹中，避免命名冲突
"""
import os
import json
import shutil
from pathlib import Path
from tqdm import tqdm

# ========== 配置路径（硬编码） ==========
KNN7_DATA_DIR = "/home/ubuntu/qwen_train/processed_data_knn"  # knn7数据源文件夹
NO_KNN7_DATA_DIR = "/home/ubuntu/qwen_train/processed_data"    # 非knn7数据文件夹（目标）
# ======================================


def merge_knn7_to_no_knn7():
    """将knn7数据合并到非knn7数据中"""
    
    knn7_dir = Path(KNN7_DATA_DIR)
    target_dir = Path(NO_KNN7_DATA_DIR)
    
    print("="*60)
    print("合并knn7数据到非knn7数据")
    print("="*60)
    print(f"源文件夹（knn7）: {knn7_dir}")
    print(f"目标文件夹（非knn7）: {target_dir}")
    print("="*60)
    
    if not knn7_dir.exists():
        print(f"错误: knn7数据文件夹不存在: {knn7_dir}")
        return
    
    if not target_dir.exists():
        print(f"错误: 目标数据文件夹不存在: {target_dir}")
        return
    
    # 统计信息
    total_folders = 0
    total_files = 0
    
    # 遍历knn7文件夹中的所有子文件夹
    all_subfolders = [f for f in knn7_dir.iterdir() if f.is_dir()]
    
    print(f"\n找到 {len(all_subfolders)} 个子文件夹需要处理\n")
    
    for subfolder in tqdm(all_subfolders, desc="处理文件夹"):
        # 新的子文件夹名称（加上_knn7后缀）
        new_subfolder_name = subfolder.name + "_knn7"
        new_subfolder_path = target_dir / new_subfolder_name
        
        # 创建新的子文件夹
        new_subfolder_path.mkdir(parents=True, exist_ok=True)
        
        # 遍历子文件夹中的所有文件
        for file in subfolder.iterdir():
            if file.is_file():
                # 获取文件扩展名
                file_stem = file.stem  # 不含扩展名的文件名
                file_suffix = file.suffix  # 扩展名
                
                # 新文件名（加上_knn7后缀）
                new_file_name = f"{file_stem}_knn7{file_suffix}"
                new_file_path = new_subfolder_path / new_file_name
                
                # 如果是JSON文件，需要修改内部的image和source_folder字段
                if file_suffix == '.json':
                    try:
                        with open(file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        # 修改image字段（加上_knn7后缀）
                        if 'image' in data:
                            old_image_name = data['image']
                            image_stem = Path(old_image_name).stem
                            image_suffix = Path(old_image_name).suffix
                            data['image'] = f"{image_stem}_knn7{image_suffix}"
                        
                        # 修改source_folder字段（加上_knn7后缀）
                        if 'source_folder' in data:
                            data['source_folder'] = data['source_folder'] + "_knn7"
                        
                        # 保存到新位置
                        with open(new_file_path, 'w', encoding='utf-8') as f:
                            json.dump(data, f, ensure_ascii=False, indent=2)
                        
                        total_files += 1
                    
                    except Exception as e:
                        print(f"\n警告: 处理JSON文件失败 {file}: {e}")
                        continue
                
                # 如果是其他文件（如图片），直接复制
                else:
                    try:
                        shutil.copy2(file, new_file_path)
                        total_files += 1
                    except Exception as e:
                        print(f"\n警告: 复制文件失败 {file}: {e}")
                        continue
        
        total_folders += 1
    
    print("\n" + "="*60)
    print("合并完成！")
    print("="*60)
    print(f"处理的文件夹数: {total_folders}")
    print(f"复制的文件数: {total_files}")
    print(f"所有knn7数据已复制到: {target_dir}")
    print("="*60)


if __name__ == '__main__':
    merge_knn7_to_no_knn7()

