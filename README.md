# 注意分词！分词影响大模型的初步发现

[论文地址(腾讯文档)V10](https://docs.qq.com/pdf/DT095ZktteVVJTE5K)

## 想法

分词是大模型第一个步骤，影响大模型训练效果、训练速度、支持上下文规模。本工作试图回答一个问题：“什么样的分词是适合大模型的分词？”。本项目核心思想是以中文为代表定性评估分词对大模型影响。按照如下步骤展开：

1. **正向构造出大模型分词错而且回答错的提示词**。发现这个非常困难，就从真歧义语句入手构造了一些大模型容易犯错的提示词，针对Qwen2.5-72B-Instruct和Deepseek-V3都可以复现。代表性提示词如下：
   
   1.1 【**李鹏飞和李鹏飞到南京了。** 请严格根据上文回答：李鹏飞在哪里？怎么到的？】，虽然Qwen和Deepseek都成了词粒度`李/鹏/飞/和/李/鹏/飞/到/南京/了`，还是没办法区分李鹏和李鹏飞是两个人。

   1.2 【**我去体育商品店里得知乒乓球拍卖完了。** 请严格根据上文回答问题：乒乓球拍还有货吗？】，分词`乒乓球/拍卖/完了`，Qwen2.5-72B-Instruct严格按照分词回答，Qwen2.5-plus可以跨过分词回答`乒乓球拍`情况。
   
3. **反向分析词表构建提示词**。根据末尾长词构建大模型容易犯错的提示词，模式相对固定，对Qwen2.5-plus、Deepseek-V3和GPT-4o都可以复现；
   
   2.1 【**最高人民法院党史学习教育需要注意的是马克思恩格斯习近平新时代中国特色社会主义思想。** 请严格根据上文回答下面问题（不要使用任何模型自身知识，如果无答案请回答不知道）：中级人民法院学习注意什么？】，黑体部分在Deepseek-V3上就分为`[最高人民法院]/[党史学习教育]/[需要注意的是]/[马克思恩格斯]/[新时代中国特色社会主义思想]...[中级人民法院]`，所以给不出正确回答。
   
   2.2 【**请问\"关注公众号天天中彩票\"** 这句话有几个汉字？分别是什么？】，黑体部分在GPT-4o中是两个词`关注/公众号天天中彩票`，所以不出正确回答。
   
   2.3 【**UseVisualStyleBackColor** 能按照英文分词吗？】，黑体部分在qwen2.5中是两个词`Use/VisualStyleBackColor`，所以不出正确回答。
   
3. 最后是想到 **根据长词让大模型定性的分析可能的来源**，初步结果如下：
   
   3.1 `GPT-4o` 中文词很长，代表性的是 **微信公众号天天中彩票**和**日本毛片免费视频观看**，中文数据非常多在线垃圾网页数据，垃圾信息或诈骗信息的关键词库、非法博彩或赌博网站的推广文本、某些低质量网站的SEO优化关键词；相反，非中文数据的确很多样，质量(可能)非常高；
   
   3.2 `Qwen2.5` 由于中文词最大长度只有4，来源数据非常多样，质量非常高; 相反，非中文数据长词代表性的是**UseVisualStyleBackColor**，可能包括跨平台开发教程、混合代码笔记、技术文章、或编程论坛的内容；
   
   3.3 `Deepseek-v3` 中文词最大长度只有16，代表性的是**习近平新时代中国特色社会主义思想**，来源自时事政治，例如报刊杂志等；非中文数据代表性词汇**pharmacological**，语料库很可能是从专业学术数据库（如PubMed、ScienceDirect等）中收集的文献资料，以及各大医学院校、研究机构的教育教学资料。

观察、总结和建议：

1. 由于大模型自身黑盒属性、长链路构建、极其复杂多样的数据，并不好分析分词和生成的关系。项目中构造的提示词提交给了三家大模型。其中qwen2.5有技术人员反馈也提到了这一点，他们认为不是分词原因，计划还是通过sft/rlhf解决。
   
2. 考虑到目前没有专门评估大模型分词的研究工作，建议可以基于现有的分词数据集，定量评估各个大模型分词效果；

3. 考虑到大模型高昂的训练成本，在不重新训练模型的前提下，现有BPE词表可以如下处理：
  
    3.1. 超过三个字的中文词都用**分词工具**重新切分，例如`习近平新时代中国特色社会主义思想=>习近平/新时代/中国/特色/社会/主义/思想`，能被拆分的词在词表中都禁用掉，例如替换成`@@原始词@@`这样不会被匹配到的格式。由于词表中的词分词再去重后数量比较少，可以人工或者让大模型评审是否合理;

    3.2. 以英文为代表的非中文，可以用大模型(例如`GPT-4o`)重新拆分，例如`UseVisualStyleBackColor=>Use/Visual/Style/Back/Color`，能被拆分的词按照上面方法禁用。拆分后的词同样可以人工评估；

4. 放弃字节级别的BPE，采用如下系统性预处理策略：

   4.1. 不使用字节级别，基于字符为基本处理粒度；
   
   4.2. 连续汉字为切分的处理单元，汉字和英文、标点、空格等都直接拆分开。连续汉字可以按照如下策略处理：

       4.2.1. 直接拆成字，字和字的关系让transformer自己学。例如deepseek-v3有35184中文词，拆成字之后只有5257个汉字。这里还有很多是生僻字。

       4.2.2. 生僻字处理，考虑到中文有8万个汉字。不同于BPE直接拆成字节码，汉字可以按照偏旁部首拆分，构成1-1映射。例如`噼`拆为`口@尸@口@辛`，具体拆分方法可以参考专家意见。生成的时候`口@尸@口@辛=>噼`。这样可以充分考虑汉字内在关系，极大减少中文字符集；
   
       4.2.3. 字合并成词，原则上到4.2.2可以拆分完训练大项目。这样会可能导致token太细模型训练效率低、支持上下文短。这里可以考虑复用词表以及根据语料库共现合并构建词，原则上词长度不超过2个字，例如 `中+国=>中国`。这里得到`{(id, '词', 词频)}`词表。

       4.2.4. 分词，由于有词表和词频，就参考传统分词方法构建分词器。其中对于有重叠的部分直接拆分，留给大模型自己学习。例如 `网球拍卖 => 网/球/拍/卖`，不处理`网球/球拍/拍卖`歧义。这个方法本身并不复杂，可以避免BPE贪婪方式导致的错误。
   
   4.3. 空格作为一个字符，不和其他字符组合；

   4.4. 非中文字符串也可以采用类似4.2.3和4.2.4的方法，当然也可以参考之前就有很多无指导的学习方法。这里也可以加上一个限制，例如词不超过5个字符。

## 记录

2025/01/14: 发现 [tokenizer-attack](https://github.com/alisawuffles/tokenizer-attack)，利用BPE分词器的合并规则列表和每个类别的示例数据，推断出训练数据中各类别的比例。和本项目关注问题切入点非常相关。一些对比：
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
