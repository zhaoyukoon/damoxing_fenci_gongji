from transformers import AutoTokenizer

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
        byte = uni_to_byte[c] if c in uni_to_byte else -1
        bs.append(byte)
    return str(bytes(bs), 'utf-8')


def segment(model, text):
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    tokens = tokenizer(text)['input_ids']
    converted = []
    id_to_tokens=tokenizer.convert_ids_to_tokens(tokens)
    t_type = type(id_to_tokens[0])
    for id in id_to_tokens:
        try:
            if type(id) is str:
                converted.append(uni_str_to_bytes(id))
            else:
                converted.append(str(id, 'utf-8'))
        except UnicodeDecodeError:
            converted.append(id)
    return converted
#converted = [str(id, 'utf-8') for id in tokenizer.convert_ids_to_tokens(tokens)]
name1 ='Qwen/Qwen-72B-Chat'
name2 ='Qwen/Qwen2.5-72B-Instruct'
name3 ='deepseek-ai/DeepSeek-V3'
text='我去体育商品店里得知乒乓球拍卖完了。请问乒乓球还有货吗？乒乓球拍呢？'
text='我去体育商品店里得知乒乓球拍卖完了。'
text='你好'
text='小明去体育商品店问：乒乓球拍卖完了吗？请问小明想买什么？'
text='李鹏飞到南京'
text='李鹏飞到南京。请问李鹏在哪里？'
for name in [name1, name2, name3]:
    seg = segment(name, text)
    print(f'{name}\t{seg}')
text='根据你提供的信息，李鹏飞已经飞到了南京，所以他现在在南京。如果你有更具体的地点或上下文信息，可以进一步明确他的位置。'
text='中国人民解放军'
text='最高人民法院党史学习教育需要注意的是马克思恩格斯习近平新时代中国特色社会主义思想'
text='生命周期函数'
for name in [name1, name2, name3]:
    seg = segment(name, text)
    print(f'{name}\t{seg}')

