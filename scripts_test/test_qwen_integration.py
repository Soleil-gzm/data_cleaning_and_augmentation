
# Qwen独立测试脚本 
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.augment_utils_add import apply_qwen_paraphrase

test_sentences = [
    "你别催了，我马上还。",
    "我已经让我朋友还了呀。",
    "我上夜班在睡觉，你晚点再打吧。",
    "这个钱不是我花的，是别人盗刷的。",
    "我不会不还的，你再给我点时间。",
    "等一下，我要还多少钱来着？你刚说的好像是五六百块钱。",
    "喂。\n喂，你好，是沈坤先生对吧。\n是我。",
    "等一下，我要查征信怎么办？会不会影响以后买房子啊。\n喂，你好，是高旭先生吗？\n嗯。"
]

for sent in test_sentences:
    result = apply_qwen_paraphrase(sent)
    print(f"原句: {sent}")
    print(f"改写: {result}")
    print("-" * 50)