import json
from tqdm import tqdm
import base64
from loguru import logger
import argparse
import jieba
from tiktoken import get_encoding
import tiktoken

from lang_detect import detect_lang
from plot_markdown import plot_length_distribution, write_lang_count_markdown

# Model list
models = [
    '360Zhinao2-7B-Chat-4K', 'Yi-1.5-34B-Chat', 'gemma-2-9b-it', 'telechat-7B',
    'Llama-3.3-70B-Instruct', 'Mistral-7B-Instruct-v0.3', 'Phi-3.5-mini-instruct',
    'deepseek_v3', 'qwen2.5-72b', 'MiniCPM3-4B', 'internlm3-8b-instruct',
    'gpt-4o', 'MiniMax-Text-01', 'glm-4-9b-chat'
]

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='词汇表转换工具')
    parser.add_argument('--tok_path', type=str, default='all',
                       choices=models + ['all'],
                       help='tokenizer路径，可选值：deepseek_v3 或 qwen2.5-72b 或者 all')
    return parser.parse_args()

def unicode_to_bytes_map():
    """Generate unicode to bytes mapping."""
    bs = (
        list(range(ord("!"), ord("~") + 1)) + 
        list(range(ord("¡"), ord("¬") + 1)) + 
        list(range(ord("®"), ord("ÿ") + 1))
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
    """Convert unicode string to bytes."""
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

def tokenize_text(text, model="gpt-4o"):
    """Tokenize text using specified model's tokenizer."""
    encoder = tiktoken.encoding_for_model(model)
    try:
        tokens = encoder.encode(text)
    except ValueError:
        logger.warning(f'ValueError for {text}')
        return [text]
    token_strings = [encoder.decode([token]) for token in tokens]
    return token_strings

def process_vocab(tok_path):
    """Process vocabulary files and generate statistics."""
    v_len = dict()
    tuples = []
    all_lens = []
    chinese_lens = []
    seg_chinese_vocab = dict()
    seg_english_vocab = dict()
    english_vocab = dict()
    seg_char_vocab = dict()
    lang_count = dict()
    vocab = []

    # Load vocabulary based on model type
    if tok_path in ['glm-4-9b-chat', '360Zhinao2-7B-Chat-4K']:
        with open(tok_path + "/tokenizer.model") as f:
            decode_error_count = 0
            for line in f:
                token, rank = line.strip().split()
                rank = int(rank)
                token = base64.b64decode(token)
                try:
                    s = str(token, 'utf-8')
                    vocab.append(s)
                except UnicodeDecodeError:
                    logger.warning(f'UnicodeDecodeError {token}')
                    s = token
                    decode_error_count += 1
            logger.error(f'decode_error_count: {decode_error_count}')

    elif tok_path == 'internlm3-8b-instruct':
        logger.info(f'load vocab from {tok_path}/vocab.txt')
        with open(tok_path+"/vocab.txt") as f:
            for line in f:
                vocab.append(line.strip())

    elif tok_path == 'gpt-4o':
        logger.info('Loading OpenAI gpt-4o tokenizer vocabulary')
        encoder = get_encoding('o200k_base')
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

    # Process vocabulary items
    for key in vocab:
        c = uni_str_to_bytes(key).replace('\n', '\\n').replace('\t', '\\t').strip()
        lang = detect_lang(c)
        lc = lang_count[lang] if lang in lang_count else 0
        lang_count[lang] = lc + 1
        segs = []

        if lang in ['pure_english', 'en', 'english']:
            english_vocab[c] = len(c)
            segs = tokenize_text(c) if len(c) > 5 else [c]
            for seg in segs:
                count = seg_english_vocab.get(seg, 0)
                seg_english_vocab[seg] = count + 1

        elif lang in ['chinese', 'zh-cn', 'zh-tw', 'zh-hk']:
            segs = list(jieba.cut(c))
            for seg in segs:
                count = seg_chinese_vocab.get(seg, 0)
                seg_chinese_vocab[seg] = count + 1
                for char in seg:
                    count = seg_char_vocab.get(char, 0)
                    seg_char_vocab[char] = count + 1

        v_len[key + "\t" + c + "\t" + ' '.join(segs)] = len(c)
        tuples.append({
            'origin': key,
            'converted': c,
            'len(converted)': len(c),
            'converted_seg': ' '.join(segs),
            'lang': lang
        })
        all_lens.append(len(c))
        if lang in ['chinese', 'zh-cn', 'zh-tw', 'zh-hk']:
            chinese_lens.append(len(c))

    # Write output files
    sorted_dict = {k: v for k, v in sorted(v_len.items(), key=lambda x: x[1])}

    logger.info(f'write to {tok_path}/vocab_extend_segged.tsv')
    with open(tok_path + '/vocab_extend_segged.tsv', 'w', encoding='utf-8') as f:
        for k, v in sorted(seg_chinese_vocab.items(), key=lambda x: -x[1]):
            f.write(f"{k}\t{v}\t{len(k)}\n")

    logger.info(f'write to {tok_path}/vocab_extend_segged_english.tsv')
    with open(tok_path + '/vocab_extend_segged_english.tsv', 'w', encoding='utf-8') as f:
        for k, v in sorted(seg_english_vocab.items(), key=lambda x: -x[1]):
            f.write(f"{k}\t{v}\t{len(k)}\n")

    logger.info(f'write to {tok_path}/vocab_english.tsv')
    with open(tok_path + '/vocab_english.tsv', 'w', encoding='utf-8') as f:
        for k, v in sorted(english_vocab.items(), key=lambda x: -x[1]):
            f.write(f"{k}\t{v}\n")

    logger.info(f'write to {tok_path}/vocab_char.tsv')
    with open(tok_path + '/vocab_char.tsv', 'w', encoding='utf-8') as f:
        for k, v in sorted(seg_char_vocab.items(), key=lambda x: -x[1]):
            f.write(f"{k}\t{v}\n")

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

def main():
    """Main program entry point."""
    args = parse_args()
    if args.tok_path == 'all':
        model_to_lang_count = dict()
        pairs = []
        for model in models:
            logger.info(f'process {model}')
            all_lens, chinese_lens, lang_count = process_vocab(model)
            pairs.append([all_lens, chinese_lens])
            model_to_lang_count[model] = lang_count
        plot_length_distribution(pairs, models)
        write_lang_count_markdown(model_to_lang_count)
    else:
        all_lens, chinese_lens = process_vocab(args.tok_path)
        plot_length_distribution([(all_lens, chinese_lens)], [args.tok_path])

if __name__ == '__main__':
    main()
