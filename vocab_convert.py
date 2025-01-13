import json
from tqdm import tqdm
import re
from loguru import logger
import argparse
import matplotlib.pyplot as plt
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='词汇表转换工具')
    parser.add_argument('--tok_path', type=str, default='both',
                       choices=['deepseek_v3', 'qwen2.5-72b'],
                       help='tokenizer路径，可选值：deepseek_v3 或 qwen2.5-72b 或者 both')
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

def process_vocab(tok_path):
    chinese_pattern = re.compile(r'^[\u4E00-\u9FFF]+$')
    v_len=dict()
    tuples = []

    logger.info(f'load vocab from {tok_path}/tokenizer.json')
    all_lens = []
    chinese_lens = []
    with open(tok_path + '/tokenizer.json') as f:
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
    logger.info(f'write to {tok_path}/vocab_extend.tsv')
    with open(tok_path + '/vocab_extend.tsv', 'w', encoding='utf-8') as f:
        for key in tqdm(sorted_dict):
            l = sorted_dict[key]
            lang='zh-cn'if  chinese_pattern.match(key.split('\t')[1]) else 'NULL'
            f.write(f'{key}\t{l}\t{lang}\n')
    logger.info(f'write to {tok_path}/vocab_extend.json')
    with open(tok_path + '/vocab_extend.json', 'w', encoding='utf-8') as f:
        json.dump(tuples, f, ensure_ascii=False, indent=4)
    return (all_lens, chinese_lens)


def plot_length_distribution(lengths_pairs, vocab_names):
    plt.figure(figsize=(10, 6))
    
    colors = ['b','c','g','k','m','r','w','y']
    # 计算直方图数据
    for i in range(len(lengths_pairs)/2):
        (all_lens, chinese_lens) = (lengths_pairs[2*i], lengths_pairs[2*i+1])
        all_counts, all_bins = np.histogram(all_lens, bins=50)
        chinese_counts, chinese_bins = np.histogram(chinese_lens, bins=50)
    
        # 获取bin中心点的位置
        all_bins_centers = (all_bins[:-1] + all_bins[1:]) / 2
        chinese_bins_centers = (chinese_bins[:-1] + chinese_bins[1:]) / 2
    
        # 使用点状线绘制
        plt.plot(all_bins_centers, all_counts, colors[2*i] + 'o--', 
                 alpha=0.7, 
                 label=vocab_names[i]+'所有词汇',
                 markersize=4,
                 linewidth=1,
                 linestyle='--'
                )
    
        plt.plot(chinese_bins_centers, chinese_counts, colors[2*i] + 'o--',
                 alpha=0.7,
                 label=vocab_names[i] + '中文词汇',
                 markersize=4,
                 linewidth=1,
                 linestyle='--'
                )
    
    # 设置对数坐标
    plt.xscale('log')
    plt.yscale('log')
    
    # 设置图表属性
    plt.title('词汇长度分布')
    plt.xlabel('长度 (log scale)')
    plt.ylabel('数量 (log scale)')
    plt.legend()
    plt.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('vocab_length_histogram.png', dpi=300)
    logger.info('Saved histogram to vocab_length_histogram.png')

if __name__ == '__main__':
    args = parse_args()
    if args.tok_path == 'both':
        (all_lens1, chinese_lens1) = process_vocab('deepseek_v3')
        (all_lens2, chinese_lens2) = process_vocab('qwen2.5-72b')
        pairs = [(all_lens1, chinese_lens1), (all_lens2, chinese_lens2)]
        plot_length_distribution(pairs, ['deepseek_v3', 'qwen2.5-72b'])
    else:
        (all_lens, chinese_lens) = process_vocab(args.tok_path)
        plot_length_distribution([(all_lens, chinese_lens)], args.tok_path)
