"""
使用vLLM启动Qwen2.5-VL推理服务
提供OpenAI兼容的API接口

优势：
1. PagedAttention优化显存
2. 连续批处理提升吞吐量
3. 更好的并发性能
4. 自动CUDA优化

启动命令：
    python start_vllm_server.py

或直接使用vllm命令：
    vllm serve /path/to/lora/adapter \
        --base-model /path/to/base/model \
        --served-model-name qwen-relation \
        --port 5003 \
        --trust-remote-code
"""
import subprocess
import yaml
from pathlib import Path


def load_config():
    """加载配置"""
    config_file = Path(__file__).parent / 'config.yaml'
    with open(config_file, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def start_vllm_server():
    """启动vLLM服务"""
    config = load_config()
    
    lora_path = config['model']['model_path']
    base_model_path = config['model']['base_model_path']
    port = 5003  # vLLM使用5003端口
    
    # 构建vllm命令
    # 注意：vLLM的LoRA支持需要先加载基础模型，然后通过--enable-lora和--lora-modules指定LoRA
    cmd = [
        'vllm', 'serve',
        base_model_path,  # 先加载基础模型
        '--served-model-name', 'qwen-relation',  # 服务模型名称
        '--port', str(port),
        '--trust-remote-code',
        '--dtype', 'float16',  # 使用FP16
        '--max-model-len', '8192',  # 最大序列长度
        '--gpu-memory-utilization', '0.9',  # GPU显存利用率
        '--enable-lora',  # 启用LoRA
        '--lora-modules', f'qwen-relation={lora_path}',  # LoRA模块：名称=路径
    ]
    
    # 如果指定了GPU
    device = config['model'].get('device')
    if device:
        import os
        if isinstance(device, str):
            gpu_id = device.split(':')[-1] if 'cuda:' in device else device
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
            print(f"🎮 GPU设置: CUDA_VISIBLE_DEVICES={gpu_id}")
    
    print("=" * 60)
    print("🚀 启动vLLM推理服务")
    print("=" * 60)
    print(f"基础模型: {base_model_path}")
    print(f"LoRA模块: qwen-relation={lora_path}")
    print(f"端口: {port}")
    print(f"服务模型名: qwen-relation")
    print("=" * 60)
    print(f"\n执行命令:\n{' '.join(cmd)}\n")
    print("=" * 60)
    print(f"启动后访问: http://localhost:{port}/v1")
    print(f"API文档: http://localhost:{port}/docs")
    print("=" * 60)
    
    # 启动服务
    subprocess.run(cmd)


if __name__ == '__main__':
    start_vllm_server()

