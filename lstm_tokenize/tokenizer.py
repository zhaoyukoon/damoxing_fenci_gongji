import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from collections import defaultdict
from loguru import logger

# 数据预处理函数
def process_string(s:str, lang:str):
    chars = []
    labels = []
    if not isinstance(s, str):
        s="nan"
        lang = "nan"
    for c in lang + s:
        if c == '-':
            if chars:  # 确保chars不为空时修改标签
                labels[-1] = 1
        else:
            chars.append(c)
            labels.append(0)
    return chars, labels

# 构建词汇表
def build_vocab(data):
    vocab = set()
    words = []
    for word in data[2]:
        words.append(word)
    langs = []
    for lang in data[4]:
        langs.append(lang)
    for index in range(len(words)):
        chars, _ = process_string(words[index], langs[index])
        vocab.update(chars)
    char_to_idx = {c: i+1 for i, c in enumerate(sorted(vocab))}  # 0留给padding
    char_to_idx['<PAD>'] = 0
    return char_to_idx

# 自定义数据集
class CharDataset(Dataset):
    def __init__(self, data, char_to_idx, max_len):
        self.data = data
        self.char_to_idx = char_to_idx
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        s = self.data.iloc[idx][2]
        lang = self.data.iloc[idx][4]
        chars, labels = process_string(s, lang)
        orig_len = len(chars)

        # 转换字符到索引
        indices = [self.char_to_idx.get(c, 0) for c in chars]

        # 生成mask
        mask = [1] * min(orig_len, self.max_len) + [0] * max(0, self.max_len - orig_len)

        # 填充/截断
        indices = indices[:self.max_len] + [0] * max(0, self.max_len - orig_len)
        labels = labels[:self.max_len] + [0] * max(0, self.max_len - orig_len)

        return (
            torch.LongTensor(indices),
            torch.FloatTensor(labels),
            torch.BoolTensor(mask)
        )

# LSTM模型
class LSTMTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        logits = self.fc(lstm_out).squeeze(-1)
        return self.sigmoid(logits)



# 保存模型
def save_model(model, path='model.pth'):
    torch.save(model.state_dict(), path)
    logger.info(f'Model saved to {path}')

# 加载模型
def load_model(model, path='model.pth'):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        logger.info(f'Model loaded from {path}')
    else:
        logger.warning(f'No model found at {path}')


# 训练和测试流程
def main():
    # 加载数据
    train_df = pd.read_csv('train.csv', header=None)
    test_df = pd.read_csv('test.csv', header=None)

    # 构建词汇表和数据集
    char_to_idx = build_vocab(train_df)
    words = [s for s in pd.concat([train_df, test_df])[2]]
    langs = [s for s in pd.concat([train_df, test_df])[4]]
    max_len = 0
    for index in range(len(words)):
        l = len(process_string(words[index], words[index])[0])
        if l > max_len:
            max_len = l
    #max_len = max(len(process_string(s)[0]) for s in pd.concat([train_df, test_df])[2])

    logger.info(f'max_len: {max_len}')
    train_dataset = CharDataset(train_df, char_to_idx, max_len)
    test_dataset = CharDataset(test_df, char_to_idx, max_len)

    # 创建DataLoader
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMTagger(
        embedding_dim=128,
        hidden_dim=256,
        vocab_size=len(char_to_idx)
    ).to(device)

    # 训练配置
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss(reduction='none')

    train = False
    if train:
        # 训练循环
        num_epochs = 10
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            total = 0

            for inputs, labels, masks in train_loader:
                inputs, labels, masks = inputs.to(device), labels.to(device), masks.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                masked_loss = (loss * masks.float()).sum()
                total_loss += masked_loss.item()
                total += masks.sum().item()

                masked_loss.backward()
                optimizer.step()

            logger.info(f'Epoch {epoch+1}, Loss: {total_loss/total:.4f}')

        save_model(model)
    else:
        logger.info(f'load model')
        load_model(model)
    # 测试循环
    model.eval()
    correct = 0
    total = 0
    whole_correct = 0
    whole_count = 0
    with torch.no_grad():
        for inputs, labels, masks in test_loader:
            inputs, labels, masks = inputs.to(device), labels.to(device), masks.to(device)
            outputs = model(inputs)
            preds = (outputs > 0.5).float()
            correct += ((preds == labels) * masks).sum().item()
            # 逐个样本判断是否正确
            batch_size = inputs.size(0)
            for i in range(batch_size):
                # 获取当前样本的预测、标签和mask
                sample_preds = preds[i][masks[i].bool()]  # 只考虑有效部分
                sample_labels = labels[i][masks[i].bool()]  # 只考虑有效部分

                # 判断当前样本是否完全正确
                if torch.equal(sample_preds, sample_labels):
                    whole_correct += 1
                whole_count += 1
            total += masks.sum().item()

    print(f'Test Accuracy: {correct/total:.4f}')
    print(f'Whole Test Accuracy: {whole_correct/whole_count:.4f}')

if __name__ == '__main__':
    main()

