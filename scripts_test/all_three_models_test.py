import os
import time
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 尝试导入 llama-cpp-python（用于 GGUF 模型）
try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False
    print("警告: llama-cpp-python 未安装，无法测试 GGUF 模型。请执行: pip install llama-cpp-python")

# ================= 配置模型路径 =================
# 请根据你的实际存放位置修改这些路径
MODELS = {
    "ERNIE-4.5-0.3B-GGUF": {
        "type": "gguf",
        "path": "/home/GUO_Zimeng/coding/data_cleaning_and_augmentation/models/ERNIE-4.5-0.3B-PT-GGUF",  # 目录路径
        # 如果目录下只有一个 .gguf 文件，脚本会自动查找；也可以直接指定完整文件路径
    },
    "Qwen3-0.6B-GGUF": {
        "type": "gguf",
        "path": "/home/GUO_Zimeng/coding/data_cleaning_and_augmentation/models/Qwen3-0.6B-Q8_0_gguf",  # 目录路径
    },
    "Qwen3-8B": {
        "type": "hf",
        "path": "/home/GUO_Zimeng/coding/data_cleaning_and_augmentation/models/Qwen3-8B",  # HuggingFace 格式目录
    }
}

# 测试句子（涵盖常见电催场景）
TEST_SENTENCES = [
    "你别催了，我马上还。",
    "我已经让我朋友还了呀。",
    "我上夜班在睡觉，你晚点再打吧。",
    "这个钱不是我花的，是别人盗刷的。",
    "我不会不还的，你再给我点时间。"
]

# ================= GGUF 模型测试函数 =================
def test_gguf_model(model_path, model_name):
    """测试 GGUF 格式模型，自动查找目录下的 .gguf 文件"""
    if not LLAMA_AVAILABLE:
        print(f"跳过 {model_name}：llama-cpp-python 未安装")
        return

    # 如果是目录，查找 .gguf 文件
    if os.path.isdir(model_path):
        gguf_files = [f for f in os.listdir(model_path) if f.endswith('.gguf')]
        if not gguf_files:
            print(f"错误：在目录 {model_path} 下未找到 .gguf 文件")
            return
        model_file = os.path.join(model_path, gguf_files[0])
    else:
        model_file = model_path

    if not os.path.exists(model_file):
        print(f"错误：模型文件不存在 {model_file}")
        return

    print(f"\n{'='*70}")
    print(f"模型: {model_name} (GGUF)")
    print(f"文件: {model_file}")
    print(f"大小: {os.path.getsize(model_file) / (1024**3):.2f} GB")
    print("加载中，请稍候...")
    try:
        llm = Llama(model_path=model_file, n_ctx=512, n_threads=4, verbose=False)
        print("加载完成\n")
    except Exception as e:
        print(f"加载失败: {e}")
        return

    for sent in TEST_SENTENCES:
        # 针对小模型优化的提示词
        prompt = f"改写：{sent}\n新句："
        start = time.time()
        try:
            output = llm(
                prompt,
                max_tokens=64,
                temperature=0.3,
                top_p=0.9,
                repeat_penalty=1.1,
                stop=["\n"],
                echo=False
            )
            elapsed = time.time() - start
            generated = output['choices'][0]['text'].strip()
            # 过滤无效输出（如下划线过多或空）
            if not generated or generated.count('_') > len(generated)/2:
                generated = "(生成无效，保留原句)"
            print(f"原句: {sent}")
            print(f"改写: {generated}")
            print(f"耗时: {elapsed:.2f} 秒")
        except Exception as e:
            print(f"推理出错: {e}")
        print("-" * 50)

# ================= HuggingFace 模型测试函数 =================
def test_hf_model(model_path, model_name):
    """测试 HuggingFace 格式模型"""
    if not os.path.exists(model_path):
        print(f"错误：模型目录不存在 {model_path}")
        return

    print(f"\n{'='*70}")
    print(f"模型: {model_name} (HuggingFace)")
    print(f"路径: {model_path}")
    print("加载中，请稍候...（可能需要几分钟，请耐心等待）")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).to('cpu')
        print("加载完成\n")
    except Exception as e:
        print(f"加载失败: {e}")
        return

    for sent in TEST_SENTENCES:
        prompt = f"改写：{sent}\n新句："
        inputs = tokenizer(prompt, return_tensors="pt").to('cpu')
        start = time.time()
        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=64,
                    do_sample=True,
                    temperature=0.3,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id
                )
            elapsed = time.time() - start
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated = generated.replace(prompt, "").strip()
            if not generated:
                generated = "(生成无效，保留原句)"
            print(f"原句: {sent}")
            print(f"改写: {generated}")
            print(f"耗时: {elapsed:.2f} 秒")
        except Exception as e:
            print(f"推理出错: {e}")
        print("-" * 50)

# ================= 主程序 =================
def main():
    print("开始测试三个模型，请耐心等待...\n")
    for model_name, config in MODELS.items():
        if config["type"] == "gguf":
            test_gguf_model(config["path"], model_name)
        elif config["type"] == "hf":
            test_hf_model(config["path"], model_name)
        else:
            print(f"未知模型类型: {config['type']}")
    print("\n所有测试完成。")

if __name__ == "__main__":
    main()