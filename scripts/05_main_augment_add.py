#!/usr/bin/env python3
"""
对话语义增强脚本（基于清洗后的 JSON）
读取最终训练数据 JSON，对每个对话中的指定轮次进行多步叠加增强，
生成多个变体对话，输出新的 JSON/JSONL 文件（保留原始数据及 loss 标记）。
"""

import json
import argparse
import random
import sys
import os
from copy import deepcopy
from pathlib import Path
from datetime import datetime

# 导入增强工具包（请确保路径正确）
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from common import augment_utils_add as aug_utils

# ========== 默认配置 ==========
DEFAULT_INPUT_ROOT = "intermediate/output_cleaning/final_training_data"
OUTPUT_ROOT = "output/augmented_data"

def get_latest_final_run_id():
    """获取 final_training_data 下最新的 *_final 目录名"""
    final_dir = Path(DEFAULT_INPUT_ROOT)
    if not final_dir.exists():
        return None
    dirs = [d for d in final_dir.iterdir() if d.is_dir() and d.name.endswith("_final")]
    if not dirs:
        return None
    dirs.sort(reverse=True)
    return dirs[0].name

def get_enhanceable_indices(messages, target_roles, only_loss_true):
    """返回可以增强的消息索引列表"""
    indices = []
    for idx, msg in enumerate(messages):
        role = msg.get("role")
        if role not in target_roles:
            continue
        content = msg.get("content", "")
        if not content.strip():
            continue
        if only_loss_true and role == "assistant" and msg.get("loss") != True:
            continue
        indices.append(idx)
    return indices

def enhance_dialogue(original_dialogue, config, rng):
    """生成变体对话列表"""
    variants = []
    messages = original_dialogue.get("messages", [])
    if not messages:
        return variants

    enhanceable = get_enhanceable_indices(messages, config["target_roles"], config["only_loss_true"])
    if not enhanceable:
        return variants

    num_variants = config["num_variants_per_dialogue"]
    if config["adaptive_variants"]:
        num_variants = max(1, min(5, len(enhanceable) // 2))

    min_turns = config["min_enhance_turns"]
    max_turns = config["max_enhance_turns"]
    aug_kwargs = config["augment_kwargs"]

    for var_id in range(num_variants):
        new_dialogue = deepcopy(original_dialogue)
        new_messages = new_dialogue["messages"]

        # 随机选择要增强的轮次数
        k = rng.randint(min_turns, max_turns)
        if k > len(enhanceable):
            k = len(enhanceable)
        selected = rng.sample(enhanceable, k)

        for idx in selected:
            original_text = new_messages[idx].get("content", "")
            if not original_text:
                continue
            variants_list = aug_utils.augment_cell_multi(original_text, **aug_kwargs)
            if variants_list:
                new_messages[idx]["content"] = variants_list[0]
        # 添加元数据
        new_dialogue["_augmented_from"] = original_dialogue.get("id", None)
        new_dialogue["_variant_id"] = var_id
        variants.append(new_dialogue)
    return variants

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_run_id", type=str, default=None,
                        help="最终训练数据的 run_id (例如 20250421_153022_clean_default_final)")
    parser.add_argument("--tag", type=str, default="default", help="增强任务标签")
    parser.add_argument("--num_variants", type=int, default=3, help="每个原始对话生成的变体数量")
    parser.add_argument("--min_turns", type=int, default=1, help="每个变体中最少增强轮次数")
    parser.add_argument("--max_turns", type=int, default=2, help="每个变体中最少增强轮次数")
    parser.add_argument("--target_roles", type=str, nargs='+', default=["user", "assistant"],
                        help="要增强的角色，可选 user/assistant")
    parser.add_argument("--only_loss_true", action="store_true",
                        help="是否只增强 loss=True 的 assistant 消息")
    parser.add_argument("--adaptive_variants", action="store_true",
                        help="根据可增强轮次数自动调整变体数量")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # 定位输入文件
    if args.source_run_id:
        input_dir = Path(DEFAULT_INPUT_ROOT) / args.source_run_id
        if not input_dir.exists():
            print(f"错误: 指定的最终数据目录不存在: {input_dir}")
            sys.exit(1)
        input_file = input_dir / "training_data.json"
    else:
        run_id = get_latest_final_run_id()
        if run_id is None:
            print("错误: 未找到最终训练数据，请先运行 04_apply_cleaned_loss_direct.py")
            sys.exit(1)
        input_file = Path(DEFAULT_INPUT_ROOT) / run_id / "training_data.json"
        print(f"自动选择最新数据: {run_id}")

    if not input_file.exists():
        print(f"错误: 输入文件不存在: {input_file}")
        sys.exit(1)

    print(f"加载原始数据: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    print(f"原始对话数量: {len(original_data)}")

    # 输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp}_augment_{args.tag}"
    output_dir = Path(OUTPUT_ROOT) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # 增强配置
    config = {
        "num_variants_per_dialogue": args.num_variants,
        "min_enhance_turns": args.min_turns,
        "max_enhance_turns": args.max_turns,
        "target_roles": args.target_roles,
        "only_loss_true": args.only_loss_true,
        "adaptive_variants": args.adaptive_variants,
        "augment_kwargs": {
            "num_variants": 1,
            "min_steps": 1,
            "max_steps": 3
        }
    }

    all_dialogues = []
    total_variants = 0
    for idx, dialogue in enumerate(original_data):
        all_dialogues.append(dialogue)  # 保留原始
        variants = enhance_dialogue(dialogue, config, rng)
        all_dialogues.extend(variants)
        total_variants += len(variants)
        if (idx+1) % 100 == 0:
            print(f"已处理 {idx+1}/{len(original_data)} 个对话，生成 {total_variants} 个变体")

    print(f"生成完成: 原始 {len(original_data)}，变体 {total_variants}，总计 {len(all_dialogues)}")

    # 保存 JSON
    output_json = output_dir / f"augmented_data_{timestamp}.json"
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(all_dialogues, f, ensure_ascii=False, indent=2)

    # 保存 JSONL
    output_jsonl = output_dir / f"augmented_data_{timestamp}.jsonl"
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for d in all_dialogues:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')

    # 保存元数据
    metadata = {
        "run_id": run_id,
        "task": "augment",
        "source_run_id": input_file.parent.name,
        "source_path": str(input_file),
        "command_line": " ".join(sys.argv),
        "config": config,
        "statistics": {
            "original_dialogues": len(original_data),
            "generated_variants": total_variants,
            "total_dialogues": len(all_dialogues)
        },
        "output_files": [str(output_json), str(output_jsonl)]
    }
    metadata_path = output_dir / "run_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"增强数据已保存至: {output_dir}")
    print(f"  JSON: {output_json}")
    print(f"  JSONL: {output_jsonl}")
    print(f"  元数据: {metadata_path}")

if __name__ == "__main__":
    main()


'''
# 基本用法（自动找最新 final 数据）
python scripts/06_augment_dialogues.py --tag v1

# 只增强 loss=True 的 assistant，生成 5 个变体，自适应
python scripts/06_augment_dialogues.py --only_loss_true --num_variants 5 --adaptive_variants --tag lossOnly_adaptive
'''