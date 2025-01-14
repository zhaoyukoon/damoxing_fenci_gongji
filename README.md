# 注意分词！分词影响大模型的初步发现

## 想法

[具体工作](https://github.com/zhaoyukoon/damoxing_fenci_gongji/blob/main/%E8%AE%BA%E6%96%87.md)

[论文地址(腾讯文档)完整版(旧)](https://docs.qq.com/pdf/DT095ZktteVVJTE5K)


## 记录

2025/01/14: 发现 [tokenizer-attack](https://github.com/alisawuffles/tokenizer-attack)，利用BPE分词器的合并规则列表和每个类别的示例数据，推断出训练数据中各类别的比例。和本项目关注问题切入点非常相关。两个工作异同分析如下：
1. `相同` 都是关注BPE分词对大模型影响;
2. `相同` 都建议谨慎使用BPE tokenizer;
3. `差异` tokenizer-attack解决如何根据BPE词表反向推到出大模型定量的数据分布，揭示模型训练数据中的一些未公开信息；本项目出发是思考分词准确率对大模型结果的影响，动机和方法都不同，虽然最后一个阶段分析数据来源是殊途同归。

2015/01/14: 给Deepseek和Qwen分别提交新的issues https://github.com/QwenLM/Qwen2.5/issues/1161 https://github.com/deepseek-ai/DeepSeek-V3/issues/273 

2015/01/14: 给OpenAI提交issue https://github.com/openai/tiktoken/issues/368

## Issue

我分别给Qwen、Deepseek和OpenAI提交了issues:
1. Qwen: https://github.com/QwenLM/Qwen2.5/issues/1159 https://github.com/QwenLM/Qwen2.5/issues/1161 ， 第一个得到回复，第二个还没有；
2. Deepseek： https://github.com/deepseek-ai/DeepSeek-V3/issues/263  https://github.com/deepseek-ai/DeepSeek-V3/issues/273 两个都尚未回复；
3. OpenAI: https://github.com/openai/tiktoken/issues/368 尚未回复；

## 提示词
`prompts.json`中包含论文中使用的提示词。其中包含了提示词、模型、调用途径、结果是否符合预期。


## 分词展示

`llm_segment.py`是对qwen、qwen2.5-72b-instruct、deepseek_v3三个大模型，给定中文输出给出分词结果。其中qwen2.5-72b-instruct、deepseek_v3的词表是unicode形式展示字节，因此会将其转化成bytes再用utf-8编码就可以正常输出。由于模型文件都是从huggingface直接加载，因此使用前需要先翻墙。

## 词表读入

`vocab_convert.py`，读入`tokener.json`处理为`vocab_extend.json`和 `vocab_extend.tsv`脚本，并生成全部词表和汉字词表长度分布。使用方法:

`python vocab_convert.py -tok_path deepseek_v3/qwen2.5-72b/both`

### `Deepseek_v3`
目录中包含如下：
1. `tokener.json`，来自 [DeepSeek-V3/tokenizer.json](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/tokenizer.json)；
2. `tokenizer_config.json`，词表配置，来自[DeepSeek-V3/tokenizer_config.json](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/tokenizer_config.json)。暂时未使用，据此可以判断出是基于LlamaTokenizerFast；
3. `vocab_extend.json`，基于tokener.json中的vocab字段将其中unicode形式转化成可读的utf-8字符串，其中包含(原始词汇、utf-8词汇、utf-8词汇长度、是否是中文)四个字段；
4. `vocab_extend.tsv`，以tsv格式展示vocab_extend.json内容，并以utf-8词汇长度顺序排序。


### `Qwen2.5-72b`
目录中包含如下：
1. `tokener.json`，来自 [Qwen2.5-72B-Instruct/tokenizer.json](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct/blob/main/tokenizer.json)；
2. `tokenizer_config.json`，词表配置，来自 [Qwen2.5-72B-Instruct/tokenizer_json.json](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct/blob/main/tokenizer_json.json)。暂时未使用，据此可以判断出是基于Qwen2Tokenizer；
3. `vocab_extend.json`，和`deepseek_v3`目录中相同格式；
4. `vocab_extend.tsv`，和`deepseek_v3`目录中相同格式；

如果您觉得本论文对您研究工作有贡献，请引用:
```
{
  "author": "赵迎功",
  "title": "注意意分词！分词影响大模型的初初步发现",
  "url": "https://github.com/zhaoyukoon/damoxing_fenci_gongji"
}
```
