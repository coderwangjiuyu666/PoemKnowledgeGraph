
import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data import Dataset, DataLoader
from torchcrf import CRF
# 读取BIO文件并构建DataFrame
def read_bio_file(file_path):
    sentences = []
    tags = []
    invalid_lines = []  # 记录无效行
    with open(file_path, encoding='utf-8') as f:
        words = []
        bio_tags = []
        line_num = 0
        for line in f:
            line_num += 1
            line = line.strip()
            if not line:
                if words:  # 空行分隔句子
                    sentences.append(words.copy())
                    tags.append(bio_tags.copy())
                    words.clear()
                    bio_tags.clear()
                continue
            # 分割字符和标签（必须有且只有一个空格）
            parts = line.split(maxsplit=1)  # 用maxsplit=1，避免标签中含空格（虽然规范标签不含）
            if len(parts) != 2:
                invalid_lines.append(f"第{line_num}行：{line}（格式错误，跳过）")
                continue
            word, tag = parts
            # 过滤异常字符（如特殊符号、空字符）
            if not word or not tag:
                invalid_lines.append(f"第{line_num}行：{line}（字符/标签为空，跳过）")
                continue
            words.append(word)
            bio_tags.append(tag)
        # 处理最后一个句子
        if words:
            sentences.append(words)
            tags.append(bio_tags)
    # 打印无效行，方便修正.bio文件
    if invalid_lines:
        print("=== 发现以下无效行（需修正.bio文件） ===")
        for msg in invalid_lines[:]:  # 打印前20个示例
            print(msg)
        print(f"共{len(invalid_lines)}行无效数据")
    return pd.DataFrame({'内容': sentences, '标签': tags})

data = read_bio_file('../entityBIO.bio')
print(data[:2])
# 划分数据集
train_data, temp_data = train_test_split(data, test_size=0.3, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=(2 / 3), random_state=42)

print('训练集数量：', len(train_data))
print('验证集数量：', len(val_data))
print('测试集数量：', len(test_data))

# 构建词汇表
all_words = set()
for text in data['内容']:
    all_words.update(text)
vocab = {"<PAD>": 0, "<UNK>": 1}
for idx, word in enumerate(all_words, 2):
    vocab[word] = idx

# 构建标签表
all_tags = set()
for tags in data['标签']:
    all_tags.update(tags)
all_tags.add("<PAD>")  # 新增Padding标签
tag_to_id = {tag: idx for idx, tag in enumerate(all_tags)}
id_to_tag = {idx: tag for tag, idx in tag_to_id.items()}
PAD_ID = tag_to_id["<PAD>"]  # Padding标签的ID

def get_pos_feature(text):
    pos_x = []
    for i, char in enumerate(text):
        is_start = 1 if i == 0 else 0  # 是否句首
        is_punc = 1 if char in [";", "，", "。", "、"] else 0  # 是否标点
        # 转成0-3的整数（is_start*2 + is_punc）
        pos = is_start * 2 + is_punc
        pos_x.append(pos)
    return pos_x

# 数据集类适配BIO格式
class PoetryBIO_Dataset(Dataset):
    def __init__(self, data, vocab, tag_to_id):
        self.data = data
        self.vocab = vocab
        self.tag_to_id = tag_to_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]['内容']  # 字符列表
        tags = self.data.iloc[idx]['标签']  # 标签列表
        text_ids = [self.vocab.get(word, self.vocab["<UNK>"]) for word in text]
        tag_ids = [self.tag_to_id[tag] for tag in tags]
        pos_x = get_pos_feature(text)  # 新增位置特征
        return torch.tensor(text_ids), torch.tensor(tag_ids), torch.tensor(pos_x)

# 修正collate_fn：标签Padding用PAD_TAG_ID
def collate_fn(batch):
    texts, tags, pos_x = zip(*batch)
    # 文本padding
    texts_padded = torch.nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=vocab["<PAD>"])
    # 标签padding
    tags_padded = torch.nn.utils.rnn.pad_sequence(tags, batch_first=True, padding_value=PAD_ID)
    # 位置特征padding（用0，对应is_start=0+is_punc=0）
    pos_padded = torch.nn.utils.rnn.pad_sequence(pos_x, batch_first=True, padding_value=0)
    mask = (texts_padded != vocab["<PAD>"])
    return texts_padded, tags_padded, pos_padded, mask

class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_size, embedding_dim=128, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.bilstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, tag_size)  # 输出每个token的标签得分
        self.crf = CRF(tag_size, batch_first=True)  # CRF层处理标签依赖

    def forward(self, x,pos_x=None,tags=None, mask=None):
        embed = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        lstm_out, _ = self.bilstm(embed)  # [batch_size, seq_len, 2*hidden_dim]
        logits = self.fc(lstm_out)  # [batch_size, seq_len, tag_size]

        if tags is not None:
            # 训练时：计算CRF损失（负对数似然）
            return -self.crf(logits, tags, mask=mask)
        else:
            # 预测时：维特比解码得到最优标签序列
            return self.crf.decode(logits, mask=mask)


# 修正训练函数的损失函数：忽略Padding标签
def train_model(model, train_loader, val_loader, optimizer, device, epochs=10):
    model.to(device)
    best_val_f1 = 0.0
    early_stop_count = 0  # 早停计数器
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch_x, batch_tags, batch_pos, mask in train_loader:
            batch_x, batch_tags, batch_pos, mask = batch_x.to(device), batch_tags.to(device), batch_pos.to(device), mask.to(device)
            optimizer.zero_grad()
            # 调用CRF模型计算损失（mask已传入，自动忽略Padding）
            loss = model(batch_x, pos_x=batch_pos, tags=batch_tags, mask=mask)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Train Loss: {total_loss / len(train_loader):.4f}")
        val_f1 = evaluate_model(model, val_loader, device)
        print(f"Epoch {epoch + 1}, Val F1: {val_f1:.4f}")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), "best_bilstm_crf.pth")
            early_stop_count = 0  # 重置计数器
        # else:
        #     early_stop_count += 1
        #     if early_stop_count >= 3:
        #         print(f"Epoch {epoch + 1}：验证集F1连续3次不提升，早停")
        #         break
    return best_val_f1

# 修正评估函数：确保只统计有效token（排除Padding）
def evaluate_model(model, data_loader, device):
    model.eval()
    all_true = []
    all_pred = []
    with torch.no_grad():
        for batch_x, batch_tags,batch_pos, mask in data_loader:
            batch_x, batch_tags, batch_pos,mask = batch_x.to(device), batch_tags.to(device),batch_pos.to(device), mask.to(device)
            # 预测时得到标签序列（List[List[int]]）
            pred_tags = model(batch_x,batch_pos, mask=mask)
            # 遍历每个样本，只取有效长度内的token
            for i in range(len(batch_x)):
                seq_len = mask[i].sum().item()
                # 真实标签：排除Padding（只取到seq_len，且不等于PAD_TAG_ID）
                true = batch_tags[i][:seq_len].cpu().numpy()
                true = true[true != PAD_ID]  # 双重保险，排除可能的Padding
                # 预测标签：长度就是seq_len（CRF解码会自动截断到有效长度）
                pred = pred_tags[i][:seq_len]
                # 只添加非Padding的真实标签和对应预测
                all_true.extend(true)
                all_pred.extend(pred)
    # 添加zero_division=0避免警告（无预测样本的标签precision设为0）
    precision = precision_score(all_true, all_pred, average="macro", zero_division=0)
    recall = recall_score(all_true, all_pred, average="macro", zero_division=0)
    f1 = f1_score(all_true, all_pred, average="macro", zero_division=0)
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    return f1


# 超参数
batch_size = 32
embedding_dim = 128
hidden_dim = 64
epochs = 30
learning_rate = 0.001

# 数据集和加载器
train_dataset = PoetryBIO_Dataset(train_data, vocab, tag_to_id)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_dataset = PoetryBIO_Dataset(val_data, vocab, tag_to_id)
val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)
test_dataset = PoetryBIO_Dataset(test_data, vocab, tag_to_id)
test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)

# 模型、优化器、设备
vocab_size = len(vocab)
tag_size = len(tag_to_id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BiLSTM_CRF(vocab_size, tag_size, embedding_dim, hidden_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 训练与评估
train_model(model, train_loader, val_loader, optimizer, device, epochs)
print("在测试集上评估模型:")
evaluate_model(model, test_loader, device)