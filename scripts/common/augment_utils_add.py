import os
import random
import re
import pandas as pd
import jieba
from nlpcda import Homophone, RandomDeleteChar,Randomword,Similarword

# 预先加载词典
jieba.initialize()
# ================= 可配置参数 =================
NUM_VARIANTS = 3                    # 每个原句生成几个变体（默认）

# 语气词库
FILLERS = ["嗯", "那个", "就是", "呃", "啊"]
TAILS = ["吧", "啊", "哦", "呗"]
TAIL_WORDS = set(["吧", "啊", "哦", "呗", "嗯", "啦", "呀", "嘛", "呐", "哈", "了", "吗", "呢"])

# # 自定义同义词映射
# SYNONYMS = {
#     "睡觉": ["休息", "睡一下"],
#     "晚点": ["晚些", "过一会儿"],
#     "打": ["联系", "打电话"],
#     "还": ["偿还", "归还"],
#     "催": ["催促", "追"],
# }

# 否定词集合（语序打乱时跳过）
NEGATION_WORDS = set(["不", "没", "无", "别", "不要", "不用", "未曾"])

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ================= 随机同义词替换器初始化 =================
SIMILARWORD_DICT_PATH = os.path.join(BASE_DIR, 'resources', 'synonyms.txt')
if os.path.exists(SIMILARWORD_DICT_PATH):
    try:
        _similarword_aug = Similarword(base_file=SIMILARWORD_DICT_PATH, create_num=3, change_rate=0.2, seed=42)
        print(f"[INFO] 成功加载自定义同义词词库: {SIMILARWORD_DICT_PATH}")
    except Exception as e:
        print(f"[ERROR] 加载自定义同义词词库失败: {e}，使用默认词库")
        _similarword_aug = Similarword(create_num=3, change_rate=0.2, seed=42)
else:
    print(f"[WARN] 自定义同义词词库不存在，使用默认词库")
    _similarword_aug = Similarword(create_num=3, change_rate=0.2, seed=42)

# ================= 同音字替换器初始化 =================
HOMOPHONE_DICT_PATH = os.path.join(BASE_DIR, 'resources', 'Homophone_tab.txt')
if not os.path.exists(HOMOPHONE_DICT_PATH):
    print(f"[WARN] 同音词词库文件不存在，将使用默认词库（可能产生生僻字）")
    _homophone_aug = Homophone(create_num=3, change_rate=0.3, seed=42)
else:
    try:
        _homophone_aug = Homophone(base_file=HOMOPHONE_DICT_PATH, create_num=3, change_rate=0.3, seed=42)
        print(f"[INFO] 成功加载同音词自定义词库: {HOMOPHONE_DICT_PATH}")
    except Exception as e:
        print(f"[ERROR] 加载同音词自定义词库失败: {e}，使用默认词库")
        _homophone_aug = Homophone(create_num=3, change_rate=0.3, seed=42)

# ================= 实体词替换器初始化 =================
ENTITY_FILE = os.path.join(BASE_DIR, 'resources', 'bank.txt')
if os.path.exists(ENTITY_FILE):
    try:
        _random_entity_aug = Randomword(base_file=ENTITY_FILE, create_num=3, change_rate=0.2, seed=42)
        print(f"[INFO] 成功加载自定义实体词库: {ENTITY_FILE}")
    except Exception as e:
        print(f"[ERROR] 加载自定义实体词库失败: {e}，使用默认词库")
        _random_entity_aug = Randomword(create_num=3, change_rate=0.2, seed=42)
else:
    print(f"[WARN] 自定义实体词库不存在，使用默认词库")
    _random_entity_aug = Randomword(create_num=3, change_rate=0.2, seed=42)

# ================= 随机删除增强器 =================
_random_delete_aug = RandomDeleteChar(create_num=3, change_rate=0.2, seed=42)
# # ================= 随机实体替换增强器（模拟不同公司/机构名称）=================
# _random_entity_aug = Randomword(create_num=3, change_rate=0.2, seed=42)

# ================= 独立增强函数（可叠加） =================

def apply_insert_filler(sentence: str) -> str:
    """插入语气词（句首或句中）"""
    filler = random.choice(FILLERS)
    if random.random() < 0.6:
        return f"{filler}，{sentence}"
    else:
        match = re.search(r'[，,。？!]', sentence)
        if match:
            pos = match.end()
            if pos > len(sentence) * 0.6:
                return f"{filler}，{sentence}"
            return sentence[:pos] + filler + "，" + sentence[pos:]
        else:
            words = sentence.split()
            if len(words) >= 2:
                return words[0] + filler + "，" + " ".join(words[1:])
            else:
                return f"{filler}，{sentence}"

def apply_stutter(sentence: str) -> str:
    """结巴模拟（重复第一个汉字）"""
    if len(sentence) < 2:
        return sentence
    match = re.search(r'[\u4e00-\u9fa5]', sentence)
    if not match:
        if len(sentence) > 1:
            return sentence[0] * 2 + sentence[1:]
        return sentence
    char = match.group()
    repeat_count = random.randint(1, 2)
    stuttered_char = char * (repeat_count + 1)
    start, end = match.start(), match.end()
    return sentence[:start] + stuttered_char + sentence[end:]

def reorder_sentence(sentence: str) -> str:
    """语序打乱（交换逗号前后，或简单谓语前置）"""
    if len(sentence) < 5:
        return sentence
    if any(neg in sentence for neg in NEGATION_WORDS):
        return sentence

    end_punct = ''
    if sentence and sentence[-1] in '。！？!?':
        end_punct = sentence[-1]
        sentence = sentence[:-1].rstrip()

    # 模式1：交换逗号前后
    if '，' in sentence:
        parts = sentence.split('，', 1)
        if len(parts) == 2 and parts[0].strip() and parts[1].strip():
            new_sent = f"{parts[1].strip()}，{parts[0].strip()}"
            return new_sent + end_punct
    if ',' in sentence:
        parts = sentence.split(',', 1)
        if len(parts) == 2 and parts[0].strip() and parts[1].strip():
            new_sent = f"{parts[1].strip()}，{parts[0].strip()}"
            return new_sent + end_punct

    # 模式2：简单谓语前置
    match = re.match(r'^(我|你)(已经|也|就|都)?(\w+?)(了|过)?(.*)$', sentence)
    if match:
        subject = match.group(1)
        adverb = match.group(2) or ''
        verb = match.group(3)
        aspect = match.group(4) or ''
        rest = match.group(5).strip()
        if verb and len(verb) >= 1:
            new_sent = f"{verb}{aspect}{rest}，{subject}{adverb}"
            new_sent = re.sub(r'\s+', '', new_sent)
            return new_sent + end_punct

    return sentence + end_punct

def apply_reorder(sentence: str) -> str:
    """语序打乱"""
    return reorder_sentence(sentence)

def homophone_augment(sentence: str) -> str:
    """同音字替换"""
    if not isinstance(sentence, str) or len(sentence.strip()) == 0:
        return sentence
    try:
        results = _homophone_aug.replace(sentence)
        if len(results) > 1:
            return random.choice(results[1:])
        else:
            return sentence
    except Exception as e:
        print(f"同音字替换出错: {e}")
        return sentence

def apply_homophone(sentence: str) -> str:
    """同音字替换（别名）"""
    return homophone_augment(sentence)

def random_delete_augment(sentence: str) -> str:
    """随机删除字符"""
    if not isinstance(sentence, str) or len(sentence.strip()) == 0:
        return sentence
    try:
        results = _random_delete_aug.replace(sentence)
        if len(results) > 1:
            return random.choice(results[1:])
        else:
            return sentence
    except Exception as e:
        print(f"随机删除字符出错: {e}")
        return sentence

def apply_random_delete(sentence: str) -> str:
    """随机删除字符（别名）"""
    return random_delete_augment(sentence)

def apply_random_entity_replace(sentence: str) -> str:
    """随机替换句子中的实体（公司/机构名称）"""
    if not isinstance(sentence, str) or len(sentence.strip()) == 0:
        return sentence
    try:
        results = _random_entity_aug.replace(sentence)
        if len(results) > 1:
            return random.choice(results[1:])
        else:
            return sentence
    except Exception as e:
        print(f"随机实体替换出错: {e}")
        return sentence
    
def apply_similarword(sentence: str) -> str:
    """使用同义词替换进行增强（替换词语为同义词）"""
    if not isinstance(sentence, str) or len(sentence.strip()) == 0:
        return sentence
    try:
        results = _similarword_aug.replace(sentence)
        if len(results) > 1:
            return random.choice(results[1:])
        else:
            return sentence
    except Exception as e:
        print(f"同义词替换出错: {e}")
        return sentence

# def apply_word_repetition(sentence: str) -> str:
#     """
#     随机重复句子中的一个**词语**（而非整个短句）
#     规则：选取长度 1~3 的中文字符串作为候选，随机重复一次
#     """
#     if not isinstance(sentence, str) or len(sentence.strip()) == 0:
#         return sentence

#     # 找出所有长度 1~3 的中文词语（连续汉字）
#     candidates = re.findall(r'[\u4e00-\u9fa5]{1,3}', sentence)
#     # 过滤掉太常见的单字（可选），保留长度 2~3 的优先，但也允许单字
#     if not candidates:
#         return sentence

#     chosen = random.choice(candidates)
#     # 只替换第一次出现
#     new_sentence = sentence.replace(chosen, chosen + chosen, 1)
#     return new_sentence

def apply_word_repetition(sentence: str) -> str:
    """使用 jieba 分词后，随机重复句子中的一个多字词语,（长度≥2），避免与 stutter 功能重叠"""
    if not isinstance(sentence, str) or len(sentence.strip()) == 0:
        return sentence

    # 使用 jieba 分词
    words = jieba.lcut(sentence)
    
    # 筛选出长度 >= 2 的词语（排除标点、单字词）
    candidates = [w for w in words if len(w) >= 2 and re.match(r'[\u4e00-\u9fa5]+', w)]
    if not candidates:
        return sentence

    chosen = random.choice(candidates)
    # 替换第一次出现的该词语
    new_sentence = sentence.replace(chosen, chosen + chosen, 1)
    return new_sentence

# ================= 多步叠加增强函数 =================

# 可用的增强函数列表（可在此处增删或调整顺序）
AUGMENT_FUNCS = [
    apply_insert_filler,
    # apply_synonym_replace,
    apply_stutter,
    apply_reorder,
    apply_homophone,
    apply_random_delete,
    apply_random_entity_replace,
    apply_similarword,
    apply_word_repetition,
]

def multi_step_augment(sentence: str, min_steps=1, max_steps=3) -> str:
    """
    对句子应用多次随机增强（可重复）
    :param sentence: 原始句子
    :param min_steps: 最少叠加次数
    :param max_steps: 最多叠加次数
    :return: 增强后的句子
    """
    if not isinstance(sentence, str) or len(sentence.strip()) == 0:
        return sentence
    steps = random.randint(min_steps, max_steps)
    result = sentence
    for _ in range(steps):
        func = random.choice(AUGMENT_FUNCS)
        result = func(result)
    return result

def augment_cell_multi(cell_value, num_variants=NUM_VARIANTS, min_steps=1, max_steps=3) -> str:
    """
    处理一个单元格（可能含 '/' 分隔的多条句子），对每条句子生成 num_variants 个变体，
    每个变体通过多次随机叠加得到。
    """
    if pd.isna(cell_value):
        return ""
    raw_sentences = [s.strip() for s in str(cell_value).split('/') if s.strip()]
    if not raw_sentences:
        return ""
    result = []
    for sent in raw_sentences:
        variants = [multi_step_augment(sent, min_steps, max_steps) for _ in range(num_variants)]
        result.append("/".join(variants))
    return "/".join(result)

# ================= 辅助函数 =================
def move_column_to_right(df, col_name, new_col_name):
    """将新列移动到原列右侧"""
    cols = df.columns.tolist()
    if new_col_name not in cols:
        return df
    idx = cols.index(col_name)
    cols.remove(new_col_name)
    cols.insert(idx + 1, new_col_name)
    return df[cols]