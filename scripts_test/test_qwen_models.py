import os
import time
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 可选：如果使用 GGUF 模型，需要 llama-cpp-python
try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False
    print("llama-cpp-python 未安装，无法测试 GGUF 模型")

# ================= 配置模型路径 =================
# 注意：这里应该是**目录**，脚本会自动查找 .gguf 文件
GGUF_MODEL_DIR = "/home/GUO_Zimeng/coding/data_cleaning_and_augmentation/models/Qwen3-0.6B-Q8_0_gguf"
# 如果目录下只有一个 .gguf 文件，会自动使用它；否则请指定完整路径

HF_MODEL_PATH = "/home/GUO_Zimeng/coding/Qwen_test/Qwen3-8B"               # 修改为实际路径

# ================= 测试句子 =================
TEST_SENTENCES = [
    "你别催了，我马上还。",
    "我已经让我朋友还了呀。",
    "我上夜班在睡觉，你晚点再打吧。"
]

def test_gguf_model(model_dir, model_name="Qwen3-0.6B"):
    """测试 GGUF 格式模型，自动查找目录下的 .gguf 文件"""
    if not LLAMA_AVAILABLE:
        print(f"跳过 {model_name}：llama-cpp-python 未安装")
        return
    if not os.path.isdir(model_dir):
        print(f"模型目录不存在: {model_dir}")
        return
    # 查找 .gguf 文件
    gguf_files = [f for f in os.listdir(model_dir) if f.endswith('.gguf')]
    if not gguf_files:
        print(f"在目录 {model_dir} 下未找到 .gguf 文件")
        return
    # 如果有多个，取第一个
    model_path = os.path.join(model_dir, gguf_files[0])
    print(f"找到模型文件: {model_path}")
    print(f"文件大小: {os.path.getsize(model_path) / (1024**3):.2f} GB")

    print(f"\n{'='*60}")
    print(f"测试模型: {model_name} (GGUF)")
    print(f"路径: {model_path}")
    try:
        # 加载模型（CPU，n_ctx 设置上下文长度，n_threads 根据 CPU 核心调整）
        print("正在加载模型，请稍候...")
        llm = Llama(model_path=model_path, n_ctx=512, n_threads=4, verbose=False)
        print("模型加载完成")
        for sent in TEST_SENTENCES:
            prompt = f"请用另一种说法改写下面的句子，保持原意不变，只输出改写后的句子：\n{sent}\n改写："
            start = time.time()
            output = llm(
                prompt,
                max_tokens=128,
                temperature=0.3,
                top_p=0.95,
                repeat_penalty=1.1,
                stop=["\n", "。", "！", "？"],
                echo=False
            )
            elapsed = time.time() - start
            generated = output['choices'][0]['text'].strip()
            print(f"原句: {sent}")
            print(f"改写: {generated if generated else '(生成失败)'}")
            print(f"耗时: {elapsed:.2f} 秒")
            print("-" * 40)
    except Exception as e:
        print(f"加载或推理失败: {e}")

def test_hf_model(model_path, model_name="Qwen3-8B"):
    """测试 HuggingFace 格式模型（原始权重）"""
    if not os.path.exists(model_path):
        print(f"模型目录不存在: {model_path}")
        return
    print(f"\n{'='*60}")
    print(f"测试模型: {model_name} (HuggingFace)")
    print(f"路径: {model_path}")
    try:
        print("正在加载模型，这可能需要几分钟并占用大量内存...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,  # CPU 上使用 float32
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).to('cpu')
        print("模型加载完成")
        for sent in TEST_SENTENCES:
            prompt = f"请用另一种说法改写下面的句子，保持原意不变，只输出改写后的句子：\n{sent}\n改写："
            inputs = tokenizer(prompt, return_tensors="pt").to('cpu')
            start = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=64,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.eos_token_id
                )
            elapsed = time.time() - start
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated = generated.replace(prompt, "").strip()
            print(f"原句: {sent}")
            print(f"改写: {generated if generated else '(生成失败)'}")
            print(f"耗时: {elapsed:.2f} 秒")
            print("-" * 40)
    except Exception as e:
        print(f"加载或推理失败: {e}")

if __name__ == "__main__":
    # 测试 GGUF 模型（轻量）
    test_gguf_model(GGUF_MODEL_DIR, "Qwen3-0.6B (GGUF)")
    # 测试 HF 模型（8B，注意内存需求）
    # test_hf_model(HF_MODEL_PATH, "Qwen3-8B (HF)")