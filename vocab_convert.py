import json
from tqdm import tqdm
import re
from loguru import logger
import argparse
import matplotlib.pyplot as plt
import numpy as np
import jieba

#jieba.enable_paddle()

def parse_args():
    parser = argparse.ArgumentParser(description='词汇表转换工具')
    parser.add_argument('--tok_path', type=str, default='all',
                       choices=['deepseek_v3', 'qwen2.5-72b', 'MiniCPM3-4B', 'internlm'],
                       help='tokenizer路径，可选值：deepseek_v3 或 qwen2.5-72b 或者 all')
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


def is_english(ch: str):
    code_point = ord(ch)
    return (
        ord('a') <= code_point <= ord('z') or
        ord('A') <= code_point <= ord('Z')
    )

chinese_pattern = re.compile(r'^[\u4E00-\u9FFF]+$')
english_pattern = re.compile(r'^[a-zA-Z]+$')
digit_pattern = re.compile(r'^[0-9]+$')
 
def detect_language(text):
    """
    Detect the probable language of a text string based on character ranges.
    Returns a dictionary with detected languages and their percentage of characters.
    
    Supports: English, Chinese, Japanese, Korean, Russian, Greek, Arabic, Hebrew, Thai
    """
    if not text:
        return {"unknown": 100.0}
    
    # Character range definitions
    ranges = {
        "english": lambda c: ord('a') <= ord(c.lower()) <= ord('z'),
        "chinese": lambda c: 0x4E00 <= ord(c) <= 0x9FFF,
        "japanese": lambda c: (0x3040 <= ord(c) <= 0x309F) or  # Hiragana
                            (0x30A0 <= ord(c) <= 0x30FF) or    # Katakana
                            (0x4E00 <= ord(c) <= 0x9FFF),      # Kanji
        "korean": lambda c: 0xAC00 <= ord(c) <= 0xD7AF,
        "russian": lambda c: 0x0400 <= ord(c) <= 0x04FF,
        "greek": lambda c: 0x0370 <= ord(c) <= 0x03FF,
        "arabic": lambda c: 0x0600 <= ord(c) <= 0x06FF,
        "hebrew": lambda c: 0x0590 <= ord(c) <= 0x05FF,
        "thai": lambda c: 0x0E00 <= ord(c) <= 0x0E7F
    }
    
    # Count characters for each language
    counts = {lang: 0 for lang in ranges}
    total_chars = 0
    
    for char in text:
        if char.isspace():
            continue
        total_chars += 1
        for lang, char_range in ranges.items():
            if char_range(char):
                counts[lang] += 1
                
    if total_chars == 0:
        return {"unknown": 100.0}
    
    # Calculate percentages and filter out languages with 0%
    percentages = {
        lang: (count / total_chars) * 100
        for lang, count in counts.items()
        if count > 0
    }
    
    # If no language detected, mark as unknown
    if not percentages:
        return {"unknown": 100.0}
    
    return percentages

def get_primary_language(text):
    """
    Returns the most probable language for the given text.
    """
    try:
        langs = detect_language(text)
    except TypeError:
        logger.error(f'detect {text} failed')
        return 'NULL'
    if not langs or langs.get("unknown", 0) == 100.0:
        return "NULL"
    return max(langs.items(), key=lambda x: x[1])[0]


def detect_lang(s):
    if '\t' in s or '\\t' in s or '\n' in s or '\\n' in s:
        return 'control'
    if chinese_pattern.match(s):
        return 'chinese'
    if digit_pattern.match(s):
        return 'digits'
    return get_primary_language(s)


def process_vocab(tok_path):
    v_len=dict()
    tuples = []

    all_lens = []
    chinese_lens = []
    seg_chinese_vocab = dict()
    english_vocab = dict()
    seg_char_vocab = dict()
    
    vocab = []
    if tok_path == 'internlm':
        logger.info(f'load vocab from {tok_path}/internlm3-8b-instruct/vocab.txt')
        tok_path=tok_path + '/internlm3-8b-instruct'
        with open(tok_path+"/vocab.txt") as f:
            for line in f:
                vocab.append(line.strip())
    else:
        logger.info(f'load vocab from {tok_path}/tokenizer.json')

        with open(tok_path + '/tokenizer.json') as f:
            j_obj = json.load(f)
            vocab = j_obj['model']['vocab']
    for key in vocab:
        c = uni_str_to_bytes(key).replace('\n', '\\n').replace('\t', '\\t')
        lang = detect_lang(c)
        segs = []
        if lang == 'english':
            english_vocab[c] = len(c)
        elif lang == 'chinese':
            segs = list(jieba.cut(c))
            for seg in segs:
                count = 0 if seg not in seg_chinese_vocab else seg_chinese_vocab[seg]
                seg_chinese_vocab[seg] = count + 1
                for char in seg:
                    count = 0 if char not in seg_char_vocab else seg_char_vocab[char]
                    seg_char_vocab[char] = count + 1
 
        v_len[key + "\t" + c + "\t" + ' '.join(segs)]=len(c)
        tuples.append({'origin': key, 'converted': c, 'len(converted)': len(c), 'converted_seg': ' '.join(segs),'lang': lang})
        all_lens.append(len(c))
        if lang == 'chinese':
            chinese_lens.append(len(c))

    sorted_dict = {key: value for key, value in sorted(
        v_len.items(), key=lambda item: item[1], reverse=False)}
    logger.info(f'write to {tok_path}/vocab_extend_segged.tsv')
    with open(tok_path + '/vocab_extend_segged.tsv', 'w', encoding='utf-8') as f:
        for k, v in sorted(seg_chinese_vocab.items(), key=lambda item: -item[1]):
            f.write(k+"\t"+str(v)+"\t"+ str(len(k)) + "\n")

    logger.info(f'write to {tok_path}/vocab_english.tsv')
    with open(tok_path + '/vocab_english.tsv', 'w', encoding='utf-8') as f:
        for k, v in sorted(english_vocab.items(), key=lambda item: -item[1]):
            f.write(k+"\t"+str(v) + "\n")


    logger.info(f'write to {tok_path}/vocab_char.tsv')
    with open(tok_path + '/vocab_char.tsv', 'w', encoding='utf-8') as f:
        for k, v in sorted(seg_char_vocab.items(), key=lambda item: -item[1]):
            f.write(k+"\t"+str(v) + "\n")
 
    logger.info(f'write to {tok_path}/vocab_extend.tsv')
    with open(tok_path + '/vocab_extend.tsv', 'w', encoding='utf-8') as f:
        for key in tqdm(sorted_dict):
            l = sorted_dict[key]
            c = key.split('\t')[1]
            lang = detect_lang(c)
            f.write(f'{key}\t{l}\t{lang}\n')
    logger.info(f'write to {tok_path}/vocab_extend.json')
    with open(tok_path + '/vocab_extend.json', 'w', encoding='utf-8') as f:
        json.dump(tuples, f, ensure_ascii=False, indent=4)
    return (all_lens, chinese_lens)


def plot_length_distribution(lengths_pairs, vocab_names):
    plt.figure(figsize=(10, 6))
    
    # Define distinct color + line style combinations
    styles = [
        {'color': '#1f77b4', 'marker': 'o', 'linestyle': '-'},  # blue, circle, solid
        {'color': '#ff7f0e', 'marker': 's', 'linestyle': '--'},  # orange, square, dashed
        {'color': '#2ca02c', 'marker': '^', 'linestyle': '-.'},  # green, triangle, dashdot
        {'color': '#d62728', 'marker': 'v', 'linestyle': ':'},   # red, triangle-down, dotted
        {'color': '#9467bd', 'marker': 'D', 'linestyle': '-'},   # purple, diamond, solid
        {'color': '#8c564b', 'marker': 'p', 'linestyle': '--'},  # brown, pentagon, dashed
        {'color': '#e377c2', 'marker': 'h', 'linestyle': '-.'},  # pink, hexagon, dashdot
        {'color': '#7f7f7f', 'marker': '*', 'linestyle': ':'}    # gray, star, dotted
    ]
    
    for i in range(len(vocab_names)):
        # Sort lengths in descending order for Zipf plot
        all_lens = sorted(lengths_pairs[i][0], reverse=True)
        chinese_lens = sorted(lengths_pairs[i][1], reverse=True)
        
        # Calculate ranks
        all_ranks = np.arange(1, len(all_lens) + 1)
        chinese_ranks = np.arange(1, len(chinese_lens) + 1)
        
        # Plot all vocabularies - with swapped axes (y=ranks, x=lengths)
        plt.plot(all_lens, all_ranks,
                color=styles[2*i]['color'],
                marker=styles[2*i]['marker'],
                linestyle=styles[2*i]['linestyle'],
                label=f"{vocab_names[i]}_all",
                alpha=0.7,
                markersize=4,
                markevery=len(all_ranks)//20)  # Plot markers at intervals
        
        # Plot Chinese vocabularies - with swapped axes
        plt.plot(chinese_lens, chinese_ranks,
                color=styles[2*i+1]['color'],
                marker=styles[2*i+1]['marker'],
                linestyle=styles[2*i+1]['linestyle'],
                label=f"{vocab_names[i]}_chinese",
                alpha=0.7,
                markersize=4,
                markevery=len(chinese_ranks)//20)  # Plot markers at intervals
    
    # Set log scales (swapped from previous version)
    plt.yscale('log')
    plt.xscale('log')
    
    # Set labels and title (swapped from previous version)
    plt.title('Vocabulary Length Zipf Distribution')
    plt.ylabel('Rank (log scale)')
    plt.xlabel('Token Length (log scale)')
    
    # Customize legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add grid with custom style
    plt.grid(True, alpha=0.3, linestyle='--', which='both')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('vocab_length_zipf.png', dpi=300, bbox_inches='tight')
    logger.info('Saved Zipf distribution plot to vocab_length_zipf.png')

if __name__ == '__main__':
    args = parse_args()
    if args.tok_path == 'all':
        choices=['deepseek_v3', 'qwen2.5-72b', 'MiniCPM3-4B', 'internlm']
        pairs= []
        for choice in choices:
            pairs.append(process_vocab(choice))
        plot_length_distribution(pairs, choices)
    else:
        (all_lens, chinese_lens) = process_vocab(args.tok_path)
        plot_length_distribution([(all_lens, chinese_lens)], [args.tok_path])
