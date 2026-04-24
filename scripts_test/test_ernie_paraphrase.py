import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.augment_utils_add import apply_model_paraphrase

test_sentences = [
    "你别催了，我马上还。",
    "我已经让我朋友还了呀。",
    "我上夜班在睡觉，你晚点再打吧。"
]

for sent in test_sentences:
    print(f"原句: {sent}")
    result = apply_model_paraphrase(sent)
    print(f"改写: {result}")
    print("-" * 50)