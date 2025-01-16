# 注意分词！分词影响大模型的初步发现

## 文章

大模型中分词使用的是数据驱动的BPE，数据集分布对词表有很大的影响。本项目分析各个大模型的词表语言和词汇长度分布。

[markdown简约版(新)](https://github.com/zhaoyukoon/damoxing_fenci_gongji/blob/main/%E8%AE%BA%E6%96%87.md)

[原始完整版(旧)](https://docs.qq.com/pdf/DT095ZktteVVJTE5K)

下面是各个大模型词表中英文和中文词汇数量统计，具体参考（[大模型词汇语言分布](https://github.com/zhaoyukoon/damoxing_fenci_gongji/blob/main/%E5%A4%A7%E6%A8%A1%E5%9E%8B%E8%AF%8D%E6%B1%87%E8%AF%AD%E8%A8%80%E5%88%86%E5%B8%83.md)和[分析报告](https://github.com/zhaoyukoon/damoxing_fenci_gongji/blob/main/%E5%A4%A7%E6%A8%A1%E5%9E%8B%E8%AF%8D%E8%A1%A8%E5%88%86%E6%9E%90%E6%8A%A5%E5%91%8A.md)）：

| 词表统计 | GPT-4o | Deepseek-v3 | Qwen2.5 |  MiniCPM3-4B  |  internlm3-8b-instruct |MiniMax-Text-01|
| ----- | ----- | ----- | ----- |  ----- |   ----- | ----- |
| 中文 | 7478 | 35184 | 24966 |  28322 |  10364 |  38420|
|  英文| 37839 | 21994 |   27376 | 15832 | 23291 | 33489 |


## 记录

2025/01/16: 发现一篇很好讲[BPE的博客](https://blog.sgdylan.com/2024/05/14/tokenizer-note/)

2025/01/16: 分析了今天刚刚发布的[MiniMax-Text01](https://huggingface.co/MiniMaxAI/MiniMax-Text-01)，中文词汇以2个和3个字为主，有非常多的长词。原始词表38421，重新分词后 23473。 另外，词表有大量日文词，最长的是`日以上更新していないブログに表示しています`。

2025/01/15: 分析了今天刚刚发布的[internlm3-8b-instruct](https://huggingface.co/internlm/internlm3-8b-instruct)，中文词汇以1个和2个字为主，和本文期望一致，不提交issue。原始词表有 10364 中文词，重新 分词去重之后有7901。

2025/01/14: 给MiniCPM提交了issue https://github.com/OpenBMB/MiniCPM/issues/276

2025/01/14: 发现 [tokenizer-attack](https://github.com/alisawuffles/tokenizer-attack)，利用BPE分词器的合并规则列表和每个类别的示例数据，推断出训练数据中各类别的比例。和本项目关注问题切入点非常相关。两个工作异同分析如下：
1. `相同` 都是关注BPE分词对大模型影响;
2. `相同` 都建议谨慎使用BPE tokenizer;
3. `差异` tokenizer-attack解决如何根据BPE词表反向推到出大模型定量的数据分布，揭示模型训练数据中的一些未公开信息；本项目出发是思考分词准确率对大模型结果的影响，动机和方法都不同，虽然最后一个阶段分析数据来源是殊途同归。

2015/01/14: 给Deepseek和Qwen分别提交新的issues https://github.com/QwenLM/Qwen2.5/issues/1161 https://github.com/deepseek-ai/DeepSeek-V3/issues/273 

2015/01/14: 给OpenAI提交issue https://github.com/openai/tiktoken/issues/368

## Issue

我分别给Qwen、Deepseek和OpenAI提交了issues:
1. Qwen: https://github.com/QwenLM/Qwen2.5/issues/1159 https://github.com/QwenLM/Qwen2.5/issues/1161 ， 都得到回复，被close；
2. Deepseek： https://github.com/deepseek-ai/DeepSeek-V3/issues/263  https://github.com/deepseek-ai/DeepSeek-V3/issues/273 两个都尚未回复；
3. OpenAI: https://github.com/openai/tiktoken/issues/368 尚未回复；
4. MiniCPM: https://github.com/OpenBMB/MiniCPM/issues/276 尚未回复；

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
