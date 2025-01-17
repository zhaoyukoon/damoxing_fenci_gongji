import json
from tqdm import tqdm
import re
from loguru import logger
import argparse
import matplotlib.pyplot as plt
import numpy as np
import jieba
from tiktoken import get_encoding
from tiktoken.model import MODEL_TO_ENCODING
import tiktoken
import base64
#jieba.enable_paddle()

models =['360Zhinao2-7B-Chat-4K','Yi-1.5-34B-Chat','gemma-2-9b-it','telechat-7B','Llama-3.3-70B-Instruct','Mistral-7B-Instruct-v0.3','Phi-3.5-mini-instruct','deepseek_v3', 'qwen2.5-72b', 'MiniCPM3-4B', 'internlm3-8b-instruct', 'gpt-4o', 'MiniMax-Text-01', 'glm-4-9b-chat']

def parse_args():
    parser = argparse.ArgumentParser(description='词汇表转换工具')
    parser.add_argument('--tok_path', type=str, default='all',
                       choices=models + ['all'],
                       help='tokenizer路径，可选值：deepseek_v3 或 qwen2.5-72b 或者 all')
    return parser.parse_args()


def tokenize_text(text, model="gpt-4o"):
    """
    Tokenize text using GPT tokenizer
    
    Args:
        text (str): Text to tokenize
        model (str): Model name to use for tokenization (default: gpt-3.5-turbo)
    
    Returns:
        list: List of tokens
        int: Number of tokens
    """
    # Get the tokenizer for the specified model
    encoder = tiktoken.encoding_for_model(model)
    
    # Encode the text to tokens
    tokens = encoder.encode(text)
    # Decode tokens back to strings for visualization
    token_strings = [encoder.decode([token]) for token in tokens]
    return token_strings

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
    try:
        if 'begin' in word:
            return word
    except TypeError:
        logger.warning(f'type error:{word}')
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
    if english_pattern.match(s):
        return 'pure_english'
    if chinese_pattern.match(s):
        return 'chinese'
    if digit_pattern.match(s):
        return 'digits'
    return get_primary_language(s)


def get_styles(style_count):
    """
    Generate a list of distinct plot styles.
    
    Args:
        style_count (int): Number of styles needed
    
    Returns:
        list: List of dictionaries containing color, linestyle, and marker configurations
    """
    # Define basic style elements
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 
             'cyan', 'magenta', 'gray', 'olive', 'pink', 'teal']
    
    linestyles = ['-', '--', '-.', ':']
    
    markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'h', '+', 'x', '|', '_']
    
    styles = []
    for i in range(style_count):
        style = {
            'color': colors[i % len(colors)],
            'linestyle': linestyles[(i // len(colors)) % len(linestyles)],
            'marker': markers[i % len(markers)]
        }
        styles.append(style)
    
    return styles

 
def process_vocab(tok_path):
    v_len=dict()
    tuples = []

    all_lens = []
    chinese_lens = []
    seg_chinese_vocab = dict()
    seg_english_vocab = dict()
    english_vocab = dict()
    seg_char_vocab = dict()
    lang_count = dict()
    vocab = []
    if tok_path in ['glm-4-9b-chat', '360Zhinao2-7B-Chat-4K']:
        with open(tok_path + "/tokenizer.model") as f:
           decode_error_count = 0
           for line in f:
               token, rank = line.strip().split()
               rank = int(rank)
               token = base64.b64decode(token)
               try:
                   s=str(token, 'utf-8')
                   vocab.append(s)
               except UnicodeDecodeError:
                   logger.warning(f'UnicodeDecodeError {token}')
                   s=token
                   decode_error_count +=1
           logger.error(f'decode_error_count: {decode_error_count}')
   
    elif tok_path == 'internlm3-8b-instruct':
        logger.info(f'load vocab from {tok_path}/vocab.txt')
        with open(tok_path+"/vocab.txt") as f:
            for line in f:
                vocab.append(line.strip())
    elif tok_path == 'gpt-4o':
        logger.info('Loading OpenAI gpt-4o tokenizer vocabulary')        
        # Add o200_base encoding
        encoder = get_encoding('o200k_base')  # Using o200_base encoding
        
        tokens = []
        for i in range(encoder.max_token_value + 1):
            try:
                token = encoder.decode([i])
                vocab.append(token)
            except:
                continue
                
        logger.info(f'Loaded {len(vocab)} tokens from o200_base vocabulary')
    else:
        logger.info(f'load vocab from {tok_path}/tokenizer.json')

        with open(tok_path + '/tokenizer.json') as f:
            j_obj = json.load(f)
            vocab = j_obj['model']['vocab']
    for key in vocab:
        c = uni_str_to_bytes(key).replace('\n', '\\n').replace('\t', '\\t')
        lang = detect_lang(c)
        lc = lang_count[lang] if lang in lang_count else 0
        lang_count[lang] = lc + 1
        segs = []
        if lang == 'pure_english':
            english_vocab[c] = len(c)
            
            segs = tokenize_text(c) if  english_pattern.match(c) and len(c) > 5 else [c]
            for seg in segs:
                count = 0 if seg not in seg_english_vocab else seg_english_vocab[seg]
                seg_english_vocab[seg] = count + 1
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

    with open(tok_path + '/vocab_extend_segged_english.tsv', 'w', encoding='utf-8') as f:
        for k, v in sorted(seg_english_vocab.items(), key=lambda item: -item[1]):
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
    return (all_lens, chinese_lens, lang_count)

def plot_length_distribution(lengths_pairs, vocab_names):
    plt.figure(figsize=(10, 6))
    # Define distinct line styles for each model
    styles = get_styles(2*len(vocab_names))
    for i in range(len(vocab_names)):
        all_lens = lengths_pairs[i][0]
        chinese_lens = lengths_pairs[i][1]
        
        # Count occurrences of each length
        all_unique_lengths, all_counts = np.unique(all_lens, return_counts=True)
        chinese_unique_lengths, chinese_counts = np.unique(chinese_lens, return_counts=True)
        
        # Plot actual length counts
        plt.plot(all_unique_lengths, 
                all_counts,
                color=styles[2*i]['color'],
                linestyle=styles[2*i]['linestyle'],
                marker=styles[2*i]['marker'],
                markersize=4,
                linewidth=1.5,
                label=f"{vocab_names[i]}_all")
        
        plt.plot(chinese_unique_lengths, 
                chinese_counts,
                color=styles[2*i+1]['color'],
                linestyle=styles[2*i+1]['linestyle'],
                marker=styles[2*i+1]['marker'],
                markersize=4,
                linewidth=1.5,
                label=f"{vocab_names[i]}_chinese")
    
    plt.xscale('log')
    plt.yscale('log')
    
    plt.title('Vocabulary Length Distribution')
    plt.xlabel('Vocabulary Length (log scale)')
    plt.ylabel('Count (log scale)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('vocab_length_histogram.png', dpi=300, bbox_inches='tight')
    logger.info('Saved histogram to vocab_length_histogram.png')

def write_lang_count_markdown(model_to_lang_count):
    """
    Write language distribution statistics to a markdown file with languages as rows
    and models as columns
    
    Args:
        model_to_lang_count (dict): Dictionary mapping model names to their language counts
    """
    logger.info('Writing language counts to model_lang_count.md')
    
    with open('大模型词汇语言分布.md', 'w', encoding='utf-8') as f:
        # Write header
        f.write('# 大模型词汇语言分布\n\n')
        
        # Get sorted list of models and languages
        models = sorted(list(model_to_lang_count.keys()))
        languages = set()
        for lang_count in model_to_lang_count.values():
            languages.update(lang_count.keys())
        languages = sorted(list(languages))
        
        # Calculate model totals
        model_totals = {}
        for model, lang_count in model_to_lang_count.items():
            model_totals[model] = sum(lang_count.values())
        
        # Write table header
        f.write('| Language | ' + ' | '.join(models) + ' |\n')
        f.write('|----------|' + '|'.join(['---' for _ in range(len(models))]) + '|\n')
        
        # Write data rows, one for each language
        for lang in languages:
            row = [lang]
            for model in models:
                count = model_to_lang_count[model].get(lang, 0)
                row.append(str(count))
            f.write('| ' + ' | '.join(row) + ' |\n')
        
        # Add total row
        total_row = ['Total']
        for model in models:
            total_row.append(str(model_totals[model]))
        f.write('| ' + ' | '.join(total_row) + ' |\n')
        
        # Add summary section
        f.write('\n## Summary\n\n')
        for model, lang_count in model_to_lang_count.items():
            f.write(f'### {model} (Total: {model_totals[model]})\n\n')
            for lang, count in sorted(lang_count.items(), key=lambda x: x[1], reverse=True):
                f.write(f'- {lang}: {count}\n')
            f.write('\n')
            
if __name__ == '__main__':
    args = parse_args()
    if args.tok_path == 'all':
        model_to_lang_count = dict()
        pairs= []
        for model in models:
            logger.info(f'process {model}')
            (all_lens, chinese_lens, lang_count) = process_vocab(model)
            pairs.append([all_lens, chinese_lens])
            model_to_lang_count[model]=lang_count
        plot_length_distribution(pairs, models)
        write_lang_count_markdown(model_to_lang_count)
    else:
        (all_lens, chinese_lens) = process_vocab(args.tok_path)
        plot_length_distribution([(all_lens, chinese_lens)], [args.tok_path])

