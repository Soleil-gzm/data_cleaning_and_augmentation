import sys
import os
import argparse
import logging
from datetime import datetime
import pandas as pd

# 将当前脚本所在目录（scripts）添加到 sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from common import augment_utils_add as aug_utils
from common.augment_utils_add import move_column_to_right

# ================= 日志配置 =================
def setup_logger(log_dir):
    """配置日志：输出到文件，控制台只显示 INFO 及以上"""
    os.makedirs(log_dir, exist_ok=True)
    log_filename = f"augment_multi_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path = os.path.join(log_dir, log_filename)
    
    logger = logging.getLogger("HumanAugmentMulti")
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.DEBUG)          # 记录所有级别到文件
    fh = logging.FileHandler(log_path, encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)               # 控制台只显示 INFO 及以上
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    logger.info(f"日志文件路径: {log_path}")
    return logger

# ================= 主流程 =================
def main():
    parser = argparse.ArgumentParser(description="多步叠加数据增强工具")
    parser.add_argument("--input", type=str, default=None,
                        help="输入Excel文件路径（若未指定，使用 data/all_data.xlsx）")
    parser.add_argument("--output", type=str, default=None,
                        help="输出Excel文件路径（若未指定，自动生成到 output_excel/ 带时间戳）")
    parser.add_argument("--sheet", type=str, default=None,
                        help="指定处理的工作表名，不指定则处理所有")
    parser.add_argument("--num_variants", type=int, default=3,
                        help="每个原句生成的变体数量（默认3）")
    parser.add_argument("--min_steps", type=int, default=1,
                        help="最少叠加次数（默认1）")
    parser.add_argument("--max_steps", type=int, default=3,
                        help="最多叠加次数（默认3）")
    args = parser.parse_args()
    
    # 确定项目根目录和文件夹
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    output_dir = os.path.join(base_dir, "output_excel")
    log_dir = os.path.join(base_dir, "logs")
    
    logger = setup_logger(log_dir)
    logger.info("=== 多步叠加增强任务开始 ===")
    logger.info(f"叠加次数范围: {args.min_steps} ~ {args.max_steps}")
    
    # 确定输入文件
    if args.input:
        input_path = args.input
        if not os.path.isabs(input_path):
            input_path = os.path.join(base_dir, input_path)
    else:
        input_path = os.path.join(data_dir, "all_data.xlsx")
        if not os.path.exists(input_path):
            input_path = os.path.join(data_dir, "test.xlsx")
    
    if not os.path.exists(input_path):
        logger.error(f"输入文件不存在: {input_path}")
        sys.exit(1)
    
    logger.info(f"输入文件: {input_path}")
    
    # 确定输出文件
    if args.output:
        output_path = args.output
        if not os.path.isabs(output_path):
            output_path = os.path.join(base_dir, output_path)
    else:
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")        # 自动生成带时间戳的输出文件名，避免覆盖
        output_filename = f"{base_name}_multi_augmented_{timestamp}.xlsx"
        output_path = os.path.join(output_dir, output_filename)
    
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"输出文件: {output_path}")
    
    # 读取 Excel
    try:
        xl = pd.ExcelFile(input_path)
        sheet_names = xl.sheet_names
        if args.sheet:
            if args.sheet not in sheet_names:
                logger.error(f"指定的工作表 '{args.sheet}' 不存在，可用: {sheet_names}")
                sys.exit(1)
            sheet_names = [args.sheet]
        logger.info(f"待处理的工作表: {sheet_names}")
    except Exception as e:
        logger.error(f"读取Excel失败: {e}")
        sys.exit(1)
    
    output_dict = {}
    for sheet_name in sheet_names:
        logger.info(f"开始处理工作表: {sheet_name}")
        df = pd.read_excel(xl, sheet_name=sheet_name)
        
        # 查找 human 列
        human_col = None
        for col in df.columns:
            col_str = str(col).strip()
            if col_str == "human(客户)" or col_str == "human" or ("human" in col_str and "客户" in col_str):
                human_col = col
                break
        if human_col is None:
            logger.warning(f"工作表 '{sheet_name}' 中未找到 human 列，跳过")
            output_dict[sheet_name] = df
            continue
        
        logger.info(f"找到 human 列: '{human_col}'，开始多步叠加增强（变体数: {args.num_variants}）")
        
        # 应用多步叠加增强
        df["human_augmented"] = df[human_col].apply(
            lambda x: aug_utils.augment_cell_multi(
                x,
                num_variants=args.num_variants,
                min_steps=args.min_steps,
                max_steps=args.max_steps
            )
        )
        df = move_column_to_right(df, human_col, "human_augmented")
        
        # 记录示例（仅DEBUG级别，不会显示在控制台）
        sample = df[[human_col, "human_augmented"]].head(2)
        for idx, row in sample.iterrows():
            logger.debug(f"  示例原句: {row[human_col]}")
            logger.debug(f"  增强后: {row['human_augmented']}")
        
        output_dict[sheet_name] = df
        logger.info(f"工作表 '{sheet_name}' 处理完成，共 {len(df)} 行")
    
    # 写入输出文件
    try:
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            for sheet_name, df in output_dict.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        logger.info(f"✅ 增强完成！结果已保存至: {output_path}")
    except Exception as e:
        logger.error(f"保存文件失败: {e}")
        sys.exit(1)
    
    # 刷新日志缓冲区
    for handler in logger.handlers:
        handler.flush()
    logger.info("任务正常结束")

if __name__ == "__main__":
    main()

'''
运行命令示例：

# 处理默认文件（data/all_data.xlsx 或 test.xlsx）
python scripts/main_augment_add.py

# 指定输入文件、叠加次数范围
python scripts/main_augment_add.py --input data/test.xlsx --min_steps 2 --max_steps 5

# 只处理特定工作表，每个原句生成5个变体
python scripts/main_augment_add.py --sheet "身份确认" --num_variants 5

'''