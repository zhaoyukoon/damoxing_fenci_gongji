import json
from tqdm import tqdm
import re
from loguru import logger
import argparse
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description='词汇表转换工具')
    parser.add_argument('--tok_path', type=str, default='deepseek_v3',
                       choices=['deepseek_v3', 'qwen2.5-72b'],
                       help='tokenizer路径，可选值：deepseek_v3 或 qwen2.5-72b')
    return parser.parse_args()

def unicode_to_bytes_map():
    """
    Returns list of utf-8 byte and a mapping to unicode strings. We specifically avoids mapping to whitespace/control
    characters the bpe code barfs on.

    The reversible bpe codes work on unicode strings. This means you need a large # of unicode characters in your vocab
    if you want to avoid UNKs. When you're at something like a 10B token dataset you end up needing around 5K for
    decent coverage. This is a significant percentage of your normal, say, 32K bpe vocab. To avoid that, we want lookup
    tables between utf-8 bytes and unicode strings.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(cs, bs))

uni_to_byte = unicode_to_bytes_map()

def uni_str_to_bytes(word):
    bs = []
    if 'begin' in word:
        return word
    for c in word:
        if c not in uni_to_byte:
            return word
        byte = uni_to_byte[c]
        bs.append(byte)
    try:
        return str(bytes(bs), 'utf-8')
    except UnicodeDecodeError:
        return word

if __name__ == '__main__':
    args = parse_args()
    chinese_pattern = re.compile(r'^[\u4E00-\u9FFF]+$')
    v_len=dict()
    tuples = []

    logger.info(f'load vocab from {args.tok_path}/tokenizer.json')
    all_lens = []
    chinese_lens = []
    with open(args.tok_path + '/tokenizer.json') as f:
        j_obj = json.load(f)
        vocab = j_obj['model']['vocab']
        for key in vocab:
            c = uni_str_to_bytes(key).replace('\n', '\\n')
            lang='zh-cn'if  chinese_pattern.match(c) else 'NULL'
            v_len[key+"\t"+c]=len(c)
            tuples.append({'origin': key, 'converted': c, 'len(converted)': len(c), 'lang': lang})
            all_lens.append(len(c))
            if lang == 'zh-cn':
                chinese_lens.append(len(c))

    sorted_dict = {key: value for key, value in sorted(
        v_len.items(), key=lambda item: item[1], reverse=False)}
    count = 1
    logger.info(f'write to {args.tok_path}/vocab_extend.tsv')
    with open(args.tok_path + '/vocab_extend.tsv', 'w', encoding='utf-8') as f:
        for key in tqdm(sorted_dict):
            l = sorted_dict[key]
            lang='zh-cn'if  chinese_pattern.match(key.split('\t')[1]) else 'NULL'
            f.write(f'{key}\t{l}\t{lang}\n')
    logger.info(f'write to {args.tok_path}/vocab_extend.json')
    with open(args.tok_path + '/vocab_extend.json', 'w', encoding='utf-8') as f:
        json.dump(tuples, f, ensure_ascii=False, indent=4)
        
    # 绘制直方图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # 绘制all_lens直方图
    ax1.hist(all_lens, bins=50, color='blue', alpha=0.7)
    ax1.set_title('All Vocab Length Distribution')
    ax1.set_xlabel('Length')
    ax1.set_ylabel('Count')
    
    # 绘制chinese_lens直方图
    ax2.hist(chinese_lens, bins=50, color='green', alpha=0.7)
    ax2.set_title('Chinese Vocab Length Distribution')
    ax2.set_xlabel('Length')
    ax2.set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('vocab_length_histogram.png')
    logger.info('Saved histogram to vocab_length_histogram.png')
