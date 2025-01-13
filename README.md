# 注意意分词！分词影响大模型的初初步发现

[论文地址(腾讯文档)](https://docs.qq.com/pdf/DT21obUxPZUh6UFdt?) ， [tokenize_llm_evaluation.v6](https://github.com/zhaoyukoon/damoxing_fenci_gongji/blob/main/tokenize_llm_evaluation.v6.pdf) 为PDF版本。
 
## 提示词
`promts.txt`中包含论文中使用的提示词。其中只有`Round0`的表示就一轮，有`Round0`和`Round1`的有两轮。提示词会持续更新。


## 分词展示

`llm_segment.py`是对qwen、qwen2.5-72b-instruct、deepseek_v3三个大模型，给定中文输出给出分词结果。其中qwen2.5-72b-instruct、deepseek_v3的词表是unicode形式展示字节，因此会将其转化成bytes再用utf-8编码就可以正常输出。由于模型文件都是从huggingface直接加载，因此使用前需要先翻墙。

## 词表读入

`vocab_convert.py`，读入`tokener.json`处理为`vocab_extend.json`和 `vocab_extend.tsv`脚本。使用方法:

`python vocab_convert.py -tok_path deepseek_v3/qwen2.5-72b`

### `deepseek_v3`
目录中包含如下：
1. `tokener.json`，来自 [DeepSeek-V3/tokenizer.json](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/tokenizer.json)；
2. `tokenizer_config.json`，词表配置，来自[DeepSeek-V3/tokenizer_config.json](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/tokenizer_config.json)。暂时未使用，据此可以判断出是基于LlamaTokenizerFast；
3. `vocab_extend.json`，基于tokener.json中的vocab字段将其中unicode形式转化成可读的utf-8字符串，其中包含(原始词汇、utf-8词汇、utf-8词汇长度、是否是中文)四个字段；
4. `vocab_extend.tsv`，以tsv格式展示vocab_extend.json内容，并以utf-8词汇长度顺序排序。


### `qwen2.5-72b`
目录中包含如下：
1. `tokener.json`，来自 [Qwen2.5-72B-Instruct/tokenizer.json](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct/blob/main/tokenizer.json)；
2. `tokenizer_config.json`，词表配置，来自 [Qwen2.5-72B-Instruct/tokenizer_json.json](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct/blob/main/tokenizer_json.json)。暂时未使用，据此可以判断出是基于Qwen2Tokenizer；
3. `vocab_extend.json`，和`deepseek_v3`目录中相同格式；
4. `vocab_extend.tsv`，和`deepseek_v3`目录中相同格式；
