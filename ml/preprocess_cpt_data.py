import os
import fitz  # PyMuPDF
import docx
import hashlib
import re
import json
from tqdm import tqdm
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- 1. 文本提取模块 ---

def extract_text_from_pdf(file_path: str) -> str:
    """从 PDF 文件中提取文本"""
    try:
        with fitz.open(file_path) as doc:
            text = "".join(page.get_text() for page in doc)
        return text
    except Exception as e:
        logging.warning(f"无法处理 PDF 文件 {file_path}: {e}")
        return ""


def extract_text_from_docx(file_path: str) -> str:
    """从 .docx 文件中提取文本"""
    try:
        doc = docx.Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text
    except Exception as e:
        logging.warning(f"无法处理 .docx 文件 {file_path}: {e}")
        return ""


# --- 2. 文本清洗模块 ---

def clean_text(text: str) -> str:
    """执行一系列文本清洗操作"""

    # 1. 替换网址和电子邮件（替换为特殊标记，而不是直接删除）
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '[URL]', text)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)

    # 2. 标准化空白字符：将多个空格、制表符替换为单个空格
    text = re.sub(r'[ \t]+', ' ', text)

    # 3. 标准化换行符：将多个连续换行符替换为单个换行符
    text = re.sub(r'[\r\n]+', '\n', text)

    # 4. 移除过短的行（通常是页眉、页脚、孤立的标题）
    min_line_length = 10  # 假设一行少于10个字符很可能是噪音
    lines = text.split('\n')
    cleaned_lines = [line.strip() for line in lines if len(line.strip()) > min_line_length]

    # 5. 重新组合文本
    text = '\n'.join(cleaned_lines)

    # 6. 去除文本开头和结尾多余的空白
    text = text.strip()

    return text


# --- 3. 主处理流程 ---

def get_text_hash(text: str) -> str:
    """计算文本的 SHA256 哈希值"""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def process_documents(input_dir: str, output_file: str, min_doc_length: int = 100):
    """
    主函数：遍历、提取、清洗、去重并保存数据。

    :param input_dir: 包含 .pdf 和 .docx 文档的输入文件夹。
    :param output_file: 输出的 .jsonl 文件路径。
    :param min_doc_length: 清洗后文档的最小字符长度（低于此值将被丢弃）。
    """

    seen_hashes = set()
    processed_files = 0
    skipped_duplicates = 0
    skipped_short = 0

    # 查找所有支持的文件
    file_paths = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.pdf', '.docx')):
                file_paths.append(os.path.join(root, file))

    logging.info(f"在 {input_dir} 中找到 {len(file_paths)} 个 .pdf/.docx 文件。")

    with open(output_file, 'w', encoding='utf-8') as f_out:
        # 使用 tqdm 显示进度条
        for file_path in tqdm(file_paths, desc="处理文档中"):
            text = ""
            if file_path.lower().endswith('.pdf'):
                text = extract_text_from_pdf(file_path)
            elif file_path.lower().endswith('.docx'):
                text = extract_text_from_docx(file_path)

            if not text:
                continue

            # 2. 清洗文本
            cleaned_text = clean_text(text)

            # 3. 质量过滤（过短）
            if len(cleaned_text) < min_doc_length:
                skipped_short += 1
                continue

            # 4. 文档级去重
            doc_hash = get_text_hash(cleaned_text)

            if doc_hash in seen_hashes:
                skipped_duplicates += 1
                continue

            # 5. 保存数据
            seen_hashes.add(doc_hash)
            output_record = {"text": cleaned_text}
            f_out.write(json.dumps(output_record, ensure_ascii=False) + '\n')
            processed_files += 1

    logging.info("--- 处理完成 ---")
    logging.info(f"总文件数: {len(file_paths)}")
    logging.info(f"成功处理并保存的文件数: {processed_files}")
    logging.info(f"因内容重复而跳过的文件数: {skipped_duplicates}")
    logging.info(f"因内容过短而跳过的文件数: {skipped_short}")
    logging.info(f"输出文件已保存至: {output_file}")


# --- 运行入口 ---

if __name__ == "__main__":
    # ----------------------------------------------------
    # TODO: 在这里修改您的输入和输出路径
    INPUT_DIRECTORY = "./手册"  # 包含您的文档的文件夹
    OUTPUT_JSONL_FILE = "./cpt_dataset.jsonl"  # 最终的数据集文件
    # ----------------------------------------------------

    if not os.path.exists(INPUT_DIRECTORY):
        logging.error(f"错误：输入目录 '{INPUT_DIRECTORY}' 不存在。")
    else:
        process_documents(INPUT_DIRECTORY, OUTPUT_JSONL_FILE)