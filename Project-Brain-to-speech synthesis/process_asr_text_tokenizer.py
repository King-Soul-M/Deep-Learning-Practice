import argparse  # 导入命令行参数解析模块
import json  # 导入JSON处理模块
import logging  # 导入日志记录模块
import os  # 导入操作系统接口模块，用于文件和路径操作
from typing import List, Optional  # 导入类型注解模块

import tokenizers  # 导入tokenizers库，用于WordPiece分词器

from nemo.collections.common.tokenizers.sentencepiece_tokenizer import create_spt_model  # 从NeMo导入SentencePiece模型创建函数
from nemo.utils.data_utils import DataStoreObject  # 导入NeMo的数据存储工具

# 创建命令行参数解析器
parser = argparse.ArgumentParser(description='Create tokenizer')
group = parser.add_mutually_exclusive_group(required=True)  # 创建互斥的命令行参数组，必须选择其中之一
group.add_argument("--manifest", default=None, type=str, help='Comma separated list of manifest files')  # 输入一个或多个manifest文件路径
group.add_argument("--data_file", default=None, help='data file from which to create tokenizer model')  # 或者输入数据文件路径
parser.add_argument("--data_root", required=True, default=None, type=str, help='Output directory')  # 指定输出目录
parser.add_argument("--vocab_size", default=1024, type=int, help='Vocabulary size')  # 词汇表大小
parser.add_argument("--tokenizer", default="wpe", choices=["spe", "wpe"], help='Type of tokenization to perform')  # 选择分词器类型（SentencePiece 或 WordPiece）
parser.add_argument(
    "--spe_type",
    default="bpe",
    choices=['bpe', 'unigram', 'char', 'word'],
    help='Type of the SentencePiece model. Can be `bpe`, `unigram`, `char` or `word`.'  # 选择SentencePiece模型类型
    'Used only if --tokenizer == `spe`',
)
parser.add_argument(
    '--spe_character_coverage',
    type=float,
    default=1.0,
    help="Character coverage percentage for SentencePiece tokenization. For languages "
    "with large vocabulary, should be close to 0.9995, otherwise kept as 1.0",  # 设置SentencePiece字符覆盖率
)
parser.add_argument('--spe_bos', action='store_true', help='Add <s> token to SentencePiece Tokenizer.')  # 是否添加BOS标记
parser.add_argument('--spe_eos', action='store_true', help='Add </s> token to SentencePiece Tokenizer.')  # 是否添加EOS标记
parser.add_argument('--spe_pad', action='store_true', help='Add <pad> token to SentencePiece Tokenizer.')  # 是否添加PAD标记
parser.add_argument(
    '--spe_user_defined_symbols', default=None, type=str, nargs='+', help='User defined symbols for SentencePiece'  # 用户定义的SentencePiece符号
)
parser.add_argument(
    '--spe_control_symbols', default=None, type=str, nargs='+', help='Control symbols for SentencePiece'  # SentencePiece控制符号
)
parser.add_argument('--spe_split_digits', action='store_true', help='Split digits into separate tokens.')  # 是否将数字分割为独立的标记

parser.add_argument(
    '--spe_sample_size',
    type=int,
    default=-1,
    help="Samples the dataset by `sample_size` if positive integer, otherwise uses whole dataset",  # 设置样本数量
)
parser.add_argument('--spe_train_extremely_large_corpus', action='store_true', help='')  # 是否训练非常大的语料库
parser.add_argument(
    '--spe_max_sentencepiece_length',
    type=int,
    default=-1,
    help='Limit the maximum number of tokens in each SentencePiece subword. '
    'Must be a positive integer > 0. By default places no limit on subword length.',  # 设置每个子词的最大长度
)
parser.add_argument(
    '--spe_no_split_by_unicode_script',
    dest='spe_split_by_unicode_script',
    action='store_false',
    help="Don't use Unicode script to split sentence pieces.",  # 是否禁止使用Unicode脚本拆分句子片段
)
parser.add_argument(
    '--spe_byte_fallback',
    dest='spe_byte_fallback',
    action='store_true',
    help="If <unk>, fallback to a byte sequence of the characters.",  # 是否启用字节回退
)
parser.add_argument('--no_lower_case', dest='lower_case', action='store_false')  # 是否禁用小写转换
parser.add_argument("--log", action='store_true')  # 是否启用日志记录
parser.set_defaults(log=False, lower_case=True, spe_train_extremely_large_corpus=False)  # 默认值
args = parser.parse_args()  # 解析命令行参数

# 从manifest文件中构建文档的函数
def __build_document_from_manifests(
    data_root: str, manifests: str,
):
    if ',' in manifests:  # 检查是否有多个manifest文件
        manifests = manifests.split(',')  # 将多个文件拆分为列表
    else:
        manifests = [manifests]  # 只有一个文件

    document_dir = os.path.join(data_root, 'text_corpus')  # 文档保存的目录
    if not os.path.exists(document_dir):  # 如果目录不存在，创建它
        os.makedirs(document_dir)

    document_path = os.path.join(document_dir, 'document.txt')  # 文档路径

    if os.path.exists(document_path):  # 如果文档已经存在，直接返回路径
        logging.info('Corpus already exists at path : %s', document_path)
        return document_path

    num_lines = 0  # 计数器，统计处理的行数
    with open(document_path, 'w') as out_writer:  # 打开文档进行写入
        for manifest in manifests:  # 遍历每个manifest文件
            with open(DataStoreObject(manifest).get(), 'r') as in_reader:  # 打开并读取manifest文件
                for line in in_reader:
                    item = json.loads(line)  # 解析每行JSON数据
                    text = item['text']  # 获取文本内容

                    out_writer.write(text + '\n')  # 将文本写入文档
                    out_writer.flush()  # 刷新缓冲区

                    num_lines += 1  # 行数加1

            logging.info(f"Finished extracting manifest : {manifest}")  # 打印日志

        logging.info("Finished extracting all manifests ! Number of sentences : {}".format(num_lines))  # 完成所有文件的处理
    return document_path  # 返回文档路径


# 处理数据并训练分词器的函数
def __process_data(
    text_path: str,
    dst_folder: str,
    vocab_size: int,
    tokenizer_type: str,
    spe_type: str,
    spe_character_coverage: float,
    spe_train_extremely_large_corpus: bool,
    spe_sample_size: int,
    spe_max_sentencepiece_length: int,
    spe_split_by_unicode_script: bool,
    spe_bos: bool,
    spe_eos: bool,
    spe_pad: bool,
    spe_control_symbols: Optional[List[str]],
    spe_user_defined_symbols: Optional[List[str]],
    spe_byte_fallback: bool,
    spe_split_digits: bool,
    lower_case: bool,
):

    if tokenizer_type == 'spe':  # 如果选择使用SentencePiece分词器
        # 准备输出目录的路径
        if spe_max_sentencepiece_length > 0:
            tokenizer_dir = os.path.join(dst_folder, 'tokenizer_{}_{}_v{}_max_{}').format(
                tokenizer_type, spe_type, vocab_size, spe_max_sentencepiece_length
            )
        else:
            tokenizer_dir = os.path.join(dst_folder, 'tokenizer_{}_{}_v{}').format(
                tokenizer_type, spe_type, vocab_size
            )

        if spe_pad:  # 如果选择添加pad标记，更新目录名
            tokenizer_dir = f'{tokenizer_dir}_pad'
        if spe_bos:  # 如果选择添加bos标记，更新目录名
            tokenizer_dir = f'{tokenizer_dir}_bos'
        if spe_eos:  # 如果选择添加eos标记，更新目录名
            tokenizer_dir = f'{tokenizer_dir}_eos'

        if not os.path.exists(tokenizer_dir):  # 如果目录不存在，创建它
            os.makedirs(tokenizer_dir)

        if os.path.exists(os.path.join(tokenizer_dir, 'tokenizer.model')):  # 如果模型已经存在，删除旧模型
            logging.warning("Model file already exists, overriding old model file !")
            os.remove(os.path.join(tokenizer_dir, 'tokenizer.model'))

        # 创建SentencePiece分词器
        tokenizer_path, vocab_path = create_spt_model(
            data_file=text_path,
            vocab_size=vocab_size,
            sample_size=spe_sample_size,
            do_lower_case=lower_case,
            output_dir=tokenizer_dir,
            tokenizer_type=spe_type,
            character_coverage=spe_character_coverage,
            train_extremely_large_corpus=spe_train_extremely_large_corpus,
            max_sentencepiece_length=spe_max_sentencepiece_length,
            split_by_unicode_script=spe_split_by_unicode_script,
            bos=spe_bos,
            eos=spe_eos,
            pad=spe_pad,
            control_symbols=spe_control_symbols,
            user_defined_symbols=spe_user_defined_symbols,
            byte_fallback=spe_byte_fallback,
            split_digits=spe_split_digits,
        )

    else:  # 如果选择使用WordPiece分词器
        tokenizer_dir = os.path.join(dst_folder, 'tokenizer_{}_v{}').format(tokenizer_type, vocab_size)

        if not os.path.exists(tokenizer_dir):  # 如果目录不存在，创建它
            os.makedirs(tokenizer_dir)

        tokenizer = tokenizers.BertWordPieceTokenizer(lowercase=lower_case)  # 创建WordPiece分词器
        tokenizer.train(text_path, vocab_size=vocab_size)  # 训练WordPiece分词器
        tokenizer.save_model(tokenizer_dir)  # 保存分词器模型

    return tokenizer_dir  # 返回保存的分词器目录


# 主程序入口
def main():
    # 解析命令行参数
    data_root = args.data_root
    manifests = args.manifest
    data_file = args.data_file
    vocab_size = args.vocab_size
    tokenizer = args.tokenizer
    spe_type = args.spe_type
    spe_character_coverage = args.spe_character_coverage
    spe_sample_size = args.spe_sample_size
    spe_train_extremely_large_corpus = args.spe_train_extremely_large_corpus
    spe_max_sentencepiece_length = args.spe_max_sentencepiece_length
    spe_split_by_unicode_script = args.spe_split_by_unicode_script
    spe_bos, spe_eos, spe_pad = args.spe_bos, args.spe_eos, args.spe_pad
    spe_control_symbols = args.spe_control_symbols
    spe_user_defined_symbols = args.spe_user_defined_symbols
    spe_byte_fallback = args.spe_byte_fallback
    spe_split_digits = args.spe_split_digits
    lower_case = args.lower_case

    if not os.path.exists(data_root):  # 如果输出目录不存在，创建它
        os.makedirs(data_root)

    if args.log:  # 如果启用日志记录，设置日志级别为INFO
        logging.basicConfig(level=logging.INFO)

    if manifests:  # 如果指定了manifest文件，调用__build_document_from_manifests函数构建文档
        text_corpus_path = __build_document_from_manifests(data_root, manifests)
    else:
        text_corpus_path = data_file # 如果没有指定manifest文件，使用data_file作为文本语料库路径

    tokenizer_path = __process_data(
        text_corpus_path,
        data_root,
        vocab_size,
        tokenizer,
        spe_type,
        lower_case=lower_case,
        spe_character_coverage=spe_character_coverage,
        spe_sample_size=spe_sample_size,
        spe_train_extremely_large_corpus=spe_train_extremely_large_corpus,
        spe_max_sentencepiece_length=spe_max_sentencepiece_length,
        spe_split_by_unicode_script=spe_split_by_unicode_script,
        spe_bos=spe_bos,
        spe_eos=spe_eos,
        spe_pad=spe_pad,
        spe_control_symbols=spe_control_symbols,
        spe_user_defined_symbols=spe_user_defined_symbols,
        spe_byte_fallback=spe_byte_fallback,
        spe_split_digits=spe_split_digits,
    )

    print("Serialized tokenizer at location :", tokenizer_path)
    logging.info('Done!')


if __name__ == "__main__":
    main()