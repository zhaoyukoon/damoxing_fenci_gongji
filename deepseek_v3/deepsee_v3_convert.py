import json
from langdetect import detect
import langdetect
from tqdm import tqdm
import re

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

v_len=dict()
tuples = []
with open('tokenizer.json') as f:
    j_obj = json.load(f)
    print('model' in j_obj)
    vocab = j_obj['model']['vocab']
    print(vocab)
    for key in vocab:
        c = uni_str_to_bytes(key)
        lang='zh-ch'if  chinese_pattern.match(converted) else 'NULL'
        v_len[key+"\t"+c]=len(c)
        tuples.append({'origin': key, 'converted': c, 'len(converted)': len(c), 'lang': lang})


sorted_dict = {key: value for key, value in sorted(
    v_len.items(), key=lambda item: item[1], reverse=False)}
count = 1
with open('deepseek_v3.vocab_extend.tsv', 'w', encoding='utf-8') as f:
    for key in tqdm(sorted_dict):
        l = sorted_dict[key]
        lang='zh-ch'if  chinese_pattern.match(converted) else 'NULL'
        f.write(f'{key}\t{l}\t{lang}\n')
with open('deepseek_v3.vocab_extend.json', 'w', encoding='utf-8') as f:
    json.dump(tuples, f, ensure_ascii=False, indent=4)
