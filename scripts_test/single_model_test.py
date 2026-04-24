#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
交互式模型测试脚本
用法：直接运行，按提示选择模型，然后选择测试模式。
"""

import os
import time
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 导入 GGUF 支持
try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False

# ================= 模型配置（请根据你的实际路径修改） =================
MODEL_CONFIGS = {
    "1": {
        "name": "Qwen3-0.6B-GGUF (Q8_0)",
        "type": "gguf",
        "path": "/home/GUO_Zimeng/coding/data_cleaning_and_augmentation/models/Qwen3-0.6B-Q8_0_gguf",
        "default_params": {
        "n_ctx": 512,
        "n_threads": 4,
        "temperature": 0.5,           # 稍微提高温度，增加变化
        "top_p": 0.9,
        "max_tokens": 48,
        "repeat_penalty": 1.1,
        "stop": ["\n"],               # 只保留换行作为停止符
        "prompt_template": "改写：{sentence}\n新句："   # 简单模板
        }
    },
    "2": {
        "name": "Qwen3-1.7B-GGUF",
        "type": "gguf",
        "path": "/home/GUO_Zimeng/coding/data_cleaning_and_augmentation/models/Qwen3-1.7B-GGUF",  # 请修改为实际路径
        "default_params": {
            "n_ctx": 1024,
            "n_threads": 4,
            "temperature": 0.3,
            "top_p": 0.9,
            "max_tokens": 80,
            "repeat_penalty": 1.1,
            "stop": ["\n"],
            "prompt_template": "改写：{sentence}\n新句："
        }
    },
    "3": {
        "name": "Qwen3-4B-GGUF",
        "type": "gguf",
        "path": "/home/GUO_Zimeng/coding/data_cleaning_and_augmentation/models/Qwen3-4B-GGUF",  # 请修改为实际路径
        "default_params": {
            "n_ctx": 2048,
            "n_threads": 4,
            "temperature": 0.3,
            "top_p": 0.9,
            "max_tokens": 96,
            "repeat_penalty": 1.1,
            "stop": ["\n"],
            "prompt_template": "改写：{sentence}\n新句："
        }
    },
}

# 预设测试句子（覆盖电催常见场景）
TEST_SENTENCES = [
    "你别催了，我马上还。",
    "我已经让我朋友还了呀。",
    "我上夜班在睡觉，你晚点再打吧。",
    "这个钱不是我花的，是别人盗刷的。",
    "我不会不还的，你再给我点时间。",
    "好的好的，我知道了。",
    "我真的没钱，你再宽限几天。",
    "钱我肯定会还，但不是现在。",
    "你打错了，我不认识这个人。",
    "我让我老婆处理一下，你晚点再打。"
]

def load_gguf_model(model_path):
    """加载 GGUF 模型，自动查找目录下的 .gguf 文件"""
    if not LLAMA_AVAILABLE:
        raise ImportError("llama-cpp-python 未安装，无法加载 GGUF 模型")
    if os.path.isdir(model_path):
        gguf_files = [f for f in os.listdir(model_path) if f.endswith('.gguf')]
        if not gguf_files:
            raise FileNotFoundError(f"在目录 {model_path} 下未找到 .gguf 文件")
        model_file = os.path.join(model_path, gguf_files[0])
    else:
        model_file = model_path
    print(f"加载 GGUF 模型: {model_file}")
    return Llama(model_path=model_file, n_ctx=512, n_threads=4, verbose=False)

def load_hf_model(model_path):
    """加载 HuggingFace 模型"""
    print(f"加载 HuggingFace 模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to('cpu')
    return tokenizer, model

def generate_gguf(llm, sentence, params):
    """使用 GGUF 模型生成改写"""
    prompt = params["prompt_template"].format(sentence=sentence)
    output = llm(
        prompt,
        max_tokens=params.get("max_tokens", 64),
        temperature=params.get("temperature", 0.3),
        top_p=params.get("top_p", 0.9),
        repeat_penalty=params.get("repeat_penalty", 1.1),
        stop=params.get("stop", ["\n"]),
        echo=False
    )
    generated = output['choices'][0]['text'].strip()
    # 简单过滤无效输出
    if not generated or generated.count('_') > len(generated)/2:
        generated = "(无效输出)"
    return generated

# def generate_gguf(llm, sentence, params):
#     prompt = params["prompt_template"].format(sentence=sentence)
#     output = llm(
#         prompt,
#         max_tokens=params.get("max_tokens", 64),
#         temperature=params.get("temperature", 0.3),
#         top_p=params.get("top_p", 0.9),
#         repeat_penalty=params.get("repeat_penalty", 1.1),
#         stop=params.get("stop", ["\n"]),
#         echo=False
#     )
#     generated = output['choices'][0]['text'].strip()
    
#     # 后处理：去除原句残留
#     if sentence in generated:
#         generated = generated.split(sentence)[0].strip()
    
#     # 简单过滤无效输出
#     if not generated or generated.count('_') > len(generated)/2:
#         generated = "(无效输出)"
#     return generated



def generate_hf(tokenizer, model, sentence, params):
    """使用 HuggingFace 模型生成改写"""
    prompt = params["prompt_template"].format(sentence=sentence)
    inputs = tokenizer(prompt, return_tensors="pt").to('cpu')
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=params.get("max_new_tokens", 64),
            do_sample=params.get("do_sample", True),
            temperature=params.get("temperature", 0.3),
            top_p=params.get("top_p", 0.9),
            repetition_penalty=params.get("repetition_penalty", 1.1),
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated = generated.replace(prompt, "").strip()
    if not generated:
        generated = "(无效输出)"
    return generated

def batch_test(generate_func, params):
    """批量测试预设句子"""
    print("\n开始批量测试...\n")
    total_time = 0
    for i, sent in enumerate(TEST_SENTENCES, 1):
        start = time.time()
        result = generate_func(sent)
        elapsed = time.time() - start
        total_time += elapsed
        print(f"{i}. 原句: {sent}")
        print(f"   改写: {result}")
        print(f"   耗时: {elapsed:.2f} 秒\n")
    print(f"总耗时: {total_time:.2f} 秒，平均: {total_time/len(TEST_SENTENCES):.2f} 秒/句")

def interactive_mode(generate_func):
    """交互模式，用户手动输入句子"""
    print("\n进入交互模式（输入 q 退出）")
    while True:
        try:
            sent = input("\n>>> ").strip()
            if sent.lower() in ('q', 'quit', 'exit'):
                break
            if not sent:
                continue
            start = time.time()
            result = generate_func(sent)
            elapsed = time.time() - start
            print(f"原句: {sent}")
            print(f"改写: {result}")
            print(f"耗时: {elapsed:.2f} 秒")
        except KeyboardInterrupt:
            print("\n退出")
            break
        except Exception as e:
            print(f"错误: {e}")

def main():
    print("="*60)
    print("模型测试工具")
    print("="*60)
    print("请选择要测试的模型：")
    for key, cfg in MODEL_CONFIGS.items():
        print(f"  {key}. {cfg['name']}")
    choice = input("请输入编号: ").strip()
    if choice not in MODEL_CONFIGS:
        print("无效选择，退出")
        sys.exit(1)
    
    config = MODEL_CONFIGS[choice]
    print(f"\n正在加载模型: {config['name']} ...")
    try:
        if config["type"] == "gguf":
            llm = load_gguf_model(config["path"])
            generate_func = lambda s: generate_gguf(llm, s, config["default_params"])
        elif config["type"] == "hf":
            tokenizer, model = load_hf_model(config["path"])
            generate_func = lambda s: generate_hf(tokenizer, model, s, config["default_params"])
        else:
            print("未知模型类型")
            sys.exit(1)
    except Exception as e:
        print(f"模型加载失败: {e}")
        sys.exit(1)
    
    print("\n模型加载完成！")
    print("请选择测试模式：")
    print("  1. 批量测试（预设句子）")
    print("  2. 交互模式（手动输入句子）")
    mode = input("请输入编号: ").strip()
    if mode == "1":
        batch_test(generate_func, config["default_params"])
    elif mode == "2":
        interactive_mode(generate_func)
    else:
        print("无效选择，退出")
    
    print("\n测试结束。")

if __name__ == "__main__":
    main()