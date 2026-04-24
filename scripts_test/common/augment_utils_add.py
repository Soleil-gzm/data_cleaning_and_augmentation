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

# 在 augment_utils_add.py 文件开头添加导入
import os
from llama_cpp import Llama

# ================= 模型加载（全局单例） =================
_llm = None
MODEL_PATH = "/home/GUO_Zimeng/.cache/modelscope/hub/models/unsloth/ERNIE-4.5-0.3B-PT-GGUF"  # 请确认实际路径，可能包含具体文件名如 'ERNIE-4.5-0.3B-PT-GGUF.Q4_K_M.gguf'

def get_llm():
    global _llm
    if _llm is None:
        # 查找目录下的 .gguf 文件
        gguf_files = [f for f in os.listdir(MODEL_PATH) if f.endswith('.gguf')]
        if not gguf_files:
            raise FileNotFoundError(f"在 {MODEL_PATH} 下未找到 .gguf 模型文件")
        model_file = os.path.join(MODEL_PATH, gguf_files[0])
        print(f"加载模型: {model_file}")
        # 使用 CPU 推理，n_ctx 上下文长度可根据需要调整，n_threads 设为 CPU 核心数
        _llm = Llama(model_path=model_file, n_ctx=512, n_threads=4, verbose=False)
        print("模型加载完成")
    return _llm

def apply_model_paraphrase(sentence: str, max_length=64) -> str:
    """使用本地 GGUF 模型进行句子语义改写"""
    if not isinstance(sentence, str) or len(sentence.strip()) == 0:
        return sentence

    llm = get_llm()
    # 构建提示词（根据不同模型微调）
    prompt = f"请用另一种说法改写下面的句子，保持原意不变，只输出改写后的句子：\n{sentence}\n改写："
    
    try:
        output = llm(
            prompt,
            max_tokens=max_length,
            temperature=0.7,
            top_p=0.9,
            repeat_penalty=1.1,
            stop=["\n", "。", "！", "？"],  # 遇到标点停止生成
            echo=False
        )
        generated = output['choices'][0]['text'].strip()
        # 如果生成结果为空或过长，则返回原句
        if generated and len(generated) < max_length * 2:
            return generated
        else:
            return sentence
    except Exception as e:
        print(f"模型改写失败: {e}")
        return sentence
    
# ================= 多步叠加增强函数 =================

# 可用的增强函数列表（可在此处增删或调整顺序）
AUGMENT_FUNCS = [
    apply_insert_filler,
    # apply_synonym_replace,
    apply_stutter,
    apply_reorder,
    apply_homophone,
    apply_random_delete,
    apply_reorder,
    apply_random_entity_replace,
    apply_similarword,
    apply_reorder,
    apply_word_repetition,
    apply_reorder,
    apply_model_paraphrase,   # 新增模型改写

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
    # 如果结果未变且句子不空，重试最多2次
    if result == sentence and len(sentence) > 1:
        for _ in range(2):
            new_result = sentence
            for _ in range(steps):
                func = random.choice(AUGMENT_FUNCS)
                new_result = func(new_result)
            if new_result != sentence:
                result = new_result
                break
    return result

def augment_cell_multi(cell_value, num_variants=NUM_VARIANTS, min_steps=1, max_steps=3, return_list=True):
    """
    处理一个单元格（可能含 '/' 分隔的多条句子），对每条句子生成 num_variants 个变体。
    返回格式：
        - return_list=True（默认）：返回列表，每个元素是一个变体字符串（若原单元格有多条句子，
          则将每条句子的变体平铺，即 len(result) == num_variants * 句子数）。
        - return_list=False：保留旧行为，返回 '/' 连接的字符串（不推荐使用）。
    
    参数：
        cell_value: 输入字符串，可能包含 '/' 分隔的多句话
        num_variants: 每句话生成的变体数量
        min_steps: 每个变体最少增强步数
        max_steps: 每个变体最多增强步数
        return_list: 是否返回列表
    """
    if pd.isna(cell_value):
        return [] if return_list else ""
    
    raw_sentences = [s.strip() for s in str(cell_value).split('/') if s.strip()]
    if not raw_sentences:
        return [] if return_list else ""
    
    # 为每个句子生成 num_variants 个变体
    all_variants = []          # 存储所有变体（平铺）
    for sent in raw_sentences:
        for _ in range(num_variants):
            variant = multi_step_augment(sent, min_steps, max_steps)
            all_variants.append(variant)
    
    if return_list:
        return all_variants
    else:
        # 旧行为：用 '/' 连接所有变体（不推荐，仅为兼容）
        return "/".join(all_variants)

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