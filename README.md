# damoxing_fenci_gongji

## 分词展示

llm_segment.py是对qwen、qwen2.5-72b-instruct、deepseek_v3三个大模型，给定中文输出给出分词结果。其中qwen2.5-72b-instruct、deepseek_v3的词表是unicode形式展示字节，因此会将其转化成bytes再用utf-8编码就可以正常输出。由于模型文件都是从huggingface直接加载，因此使用前需要先翻墙。

## 词表读入

deepseek_v3 目录中包含如下：
1. tokener.json，包含词表，来自 https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/tokenizer.json；
2. tokenizer_config.json，词表配置，来自 https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/tokenizer_config.json。暂时未使用，据此可以判断出是基于LlamaTokenizerFast；
3. deepseek_v3.vocab_extend.json，基于tokener.json中的vocab字段将其中unicode形式转化成可读的utf-8字符串，其中包含(原始词汇、utf-8词汇、utf-8词汇长度、是否是中文)四个字段；
4. deepseek_v3.vocab_extend.tsv，以tsv格式展示deepseek_v3.vocab_extend.json内容，并以utf-8词汇长度顺序排序。由于utf-8词汇有换行，需要再处理一下。
5. deepsee_v3_convert.py，读入tokener.json处理为deepseek_v3.vocab_extend.json和 deepseek_v3.vocab_extend.tsv脚本。


qwen2.5-72b 目录中包含如下：
1. tokener.json，包含词表，来自 https://huggingface.co/Qwen/Qwen2.5-72B-Instruct/blob/main/tokenizer.json；
2. tokenizer_config.json，词表配置，来自 https://huggingface.co/Qwen/Qwen2.5-72B-Instruct/blob/main/tokenizer_json.json。暂时未使用，据此可以判断出是基于Qwen2Tokenizer；
3. qwen2.5.vocab_extend.json，基于tokener.json中的vocab字段将其中unicode形式转化成可读的utf-8字符串，其中包含(原始词汇、utf-8词汇、utf-8词汇长度、是否是中文)四个字段；
4. qwen2.5.vocab_extend.tsv，以tsv格式展示qwen2.5.vocab_extend.json内容，并以utf-8词汇长度顺序排序。由于utf-8词汇有换行，需要再处理一下。
5. qwen2.5_convert.py，读入tokener.json处理为qwen2.5.vocab_extend.json和 qwen2.5.vocab_extend.tsv脚本。
