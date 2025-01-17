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
    if -1 in  bs:
        return word
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
text='根据你提供的信息，李鹏飞已经飞到了南京，所以他现在在南京。如果你有更具体的地点或上下文信息，可以进一步明确他的位置。'
text='中国人民解放军'
text='最高人民法院党史学习教育需要注意的是马克思恩格斯习近平新时代中国特色社会主义思想'
text='生命周期函数'
text='李鹏飞和李鹏飞到南京了。请严格根据上文回答：李鹏在哪里？怎么到的？'

from openai import OpenAI

client = OpenAI(api_key="sk-cc7f408684f24ed786cc9c70784beed6", base_url="https://api.deepseek.com")

prompt= "给定一段绳子，随机切两刀，切完之后的三段绳子构成三角形概率有多大？"
prompt = '中国人民解放军创立于1935年。请根据上述语句回答解放军哪里成立？'
prompt='最高人民法院党史学习教育需要注意的是马克思恩格斯习近平新时代中国特色社会主义思想。请告诉我的语句有几个字？'
prompt='习近平新时代中国特色社会主义思想。请严格根据上文回答下面问题（不要使用任何模型自身知识）：上文中提到了思想在哪个国家？'

prompt='最高人民法院党史学习教育需要注意的是马克思恩格斯习近平新时代中国特色社会主义思想。请严格根据上文回答下面问题（不要使用任何模型自身知识）：中国人民银行在学习什么？'
prompt='习近平新时代中国特色社会主义思想。请严格尝试对上面语句拆成字。'
prompt='最高人民法院党史学习教育需要注意的是马克思恩格斯习近平新时代中国特色社会主义思想。请对上面语句拆成字'
prompt = 'segment .translatesAutoresizingMaskIntoConstraints to subwords'
seg = segment(name3, prompt)
'''
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": prompt},
    ],
    stream=False
)

print(f'prompt: {prompt}')
print(f'segment: {seg}')
print(f'deepseek-v3:{response.choices[0].message.content}')
'''

text = 'Unigram 语言建模首先在 《 Improving neural network translation models with multiple subword candidates》 中提出。这种方法与 WordPiece 相同点是：同样使用语言模型来挑选子词。与 WordPiece 最大区别：WordPiece 算法的词表大小都是从小到大变化。UniLM 的词库则是从大到小变化,即先初始化一个大词表，根据评估准则不断丢弃词表，直到满足限定条件。ULM 算法考虑了句子的不同分词可能，因而能够输出带概率的多个子词分段。'
text = '网球拍卖了'
text = '球拍'
text = '清水出芙蓉'
models=['openbmb/MiniCPM3-4B','deepseek-ai/DeepSeek-V3', 'Qwen/Qwen2.5-72B-Instruct', 'MiniMaxAI/MiniMax-Text-01', 'Qwen/Qwen2.5-72B-Instruct']
for name in models:
    seg = segment(name, text)
    print(f'{name}\t{seg}')

