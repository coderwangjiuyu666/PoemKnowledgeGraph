import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import warnings

warnings.filterwarnings('ignore')


def preprocess_bio_data(bio_path, max_dist=50):
    """数据预处理：解析BIO文件，构建实体对样本"""
    # 读取BIO标注文件
    bio_data = []
    with open(bio_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            token, label = line.split()
            bio_data.append((token, label))

    # 分割文本片段（每首诗+作者+朝代为一个片段）
    segments = []
    current_segment = {'poem': [], 'author': [], 'dynasty': []}
    current_entity_type = None

    for token, label in bio_data:
        if label == 'B-Poem':
            if any(current_segment.values()):
                segments.append(current_segment)
                current_segment = {'poem': [], 'author': [], 'dynasty': []}
            current_entity_type = 'poem'
            current_segment['poem'].append(token)
        elif label == 'I-Poem' and current_entity_type == 'poem':
            current_segment['poem'].append(token)
        elif label == 'B-Author':
            current_entity_type = 'author'
            current_segment['author'].append(token)
        elif label == 'I-Author' and current_entity_type == 'author':
            current_segment['author'].append(token)
        elif label == 'B-Dynasty':
            current_entity_type = 'dynasty'
            current_segment['dynasty'].append(token)
        elif label == 'I-Dynasty' and current_entity_type == 'dynasty':
            current_segment['dynasty'].append(token)

    if any(current_segment.values()):
        segments.append(current_segment)

    # 构建关系样本
    samples = []
    for seg in segments:
        poem_tokens = seg['poem']
        author_tokens = seg['author']
        dynasty_tokens = seg['dynasty']

        if not author_tokens or (not poem_tokens and not dynasty_tokens):
            continue

        context_tokens = poem_tokens + author_tokens + dynasty_tokens
        poem_len = len(poem_tokens)
        author_len = len(author_tokens)
        dynasty_len = len(dynasty_tokens)

        # 作者-朝代（属于）
        if dynasty_tokens:
            e1_start = poem_len
            e1_end = e1_start + author_len - 1
            e2_start = e1_end + 1
            e2_end = e2_start + dynasty_len - 1
            samples.append((context_tokens, e1_start, e1_end, e2_start, e2_end, 1))

        # 作者-诗歌（创作）
        if poem_tokens:
            e1_start = poem_len
            e1_end = e1_start + author_len - 1
            e2_start = 0
            e2_end = poem_len - 1
            samples.append((context_tokens, e1_start, e1_end, e2_start, e2_end, 2))

        # 诗歌-朝代（无关系）
        if poem_tokens and dynasty_tokens:
            e1_start = 0
            e1_end = poem_len - 1
            e2_start = poem_len + author_len
            e2_end = e2_start + dynasty_len - 1
            samples.append((context_tokens, e1_start, e1_end, e2_start, e2_end, 0))

    # 构建词表
    word_counts = Counter()
    for sample in samples:
        word_counts.update(sample[0])

    word_to_idx = {'<PAD>': 0, '<UNK>': 1}
    for word, _ in word_counts.most_common():
        if word not in word_to_idx:
            word_to_idx[word] = len(word_to_idx)
    vocab_size = len(word_to_idx)

    # 最大序列长度
    max_seq_len = max(len(sample[0]) for sample in samples) if samples else 128

    # 样本编码
    def encode_sample(sample):
        context_tokens, e1_s, e1_e, e2_s, e2_e, relation = sample

        # 词编码
        x = [word_to_idx.get(token, 1) for token in context_tokens]
        if len(x) < max_seq_len:
            x += [0] * (max_seq_len - len(x))
        else:
            x = x[:max_seq_len]
            e1_s, e1_e = min(e1_s, max_seq_len - 1), min(e1_e, max_seq_len - 1)
            e2_s, e2_e = min(e2_s, max_seq_len - 1), min(e2_e, max_seq_len - 1)

        # 实体中心位置
        head_center = (e1_s + e1_e) // 2
        tail_center = (e2_s + e2_e) // 2

        # 相对位置编码
        head_pos = []
        tail_pos = []
        for i in range(max_seq_len):
            d_h = min(max(i - head_center, -max_dist), max_dist) + max_dist
            d_t = min(max(i - tail_center, -max_dist), max_dist) + max_dist
            head_pos.append(d_h)
            tail_pos.append(d_t)

        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(head_pos, dtype=torch.long),
            torch.tensor(tail_pos, dtype=torch.long),
            torch.tensor(relation, dtype=torch.long)
        )

    # 划分数据集
    train_samples, val_test_samples = train_test_split(samples, test_size=0.3, random_state=42)
    val_samples, test_samples = train_test_split(val_test_samples, test_size=0.5, random_state=42)

    # 数据集类
    class RelationDataset(data.Dataset):
        def __init__(self, samples, encode_fn):
            self.samples = samples
            self.encode_fn = encode_fn

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            return self.encode_fn(self.samples[idx])

    train_dataset = RelationDataset(train_samples, encode_sample)
    val_dataset = RelationDataset(val_samples, encode_sample)
    test_dataset = RelationDataset(test_samples, encode_sample)

    return (
        train_dataset, val_dataset, test_dataset,
        word_to_idx, vocab_size, max_seq_len, max_dist,
        train_samples, val_samples, test_samples  # 返回分割后的样本列表
    )


class AttentionLayer(nn.Module):
    """注意力层：计算序列中每个位置的权重"""

    def __init__(self, hidden_dim):
        super().__init__()
        self.w = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_output):
        attn_weights = self.w(lstm_output).squeeze(-1)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_output).squeeze(1)
        return context, attn_weights


class AttBLSTMRelationExtractor(nn.Module):
    def __init__(
            self, vocab_size, embed_dim=100, hidden_dim=128,
            num_layers=2, num_classes=3, max_dist=50,
            dropout=0.5, bidirectional=True
    ):
        super().__init__()

        # 词嵌入层
        self.word_embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=0
        )

        # 位置嵌入层
        pos_embed_dim = embed_dim // 2
        self.head_pos_embedding = nn.Embedding(
            2 * max_dist + 1, pos_embed_dim, padding_idx=0
        )
        self.tail_pos_embedding = nn.Embedding(
            2 * max_dist + 1, pos_embed_dim, padding_idx=0
        )

        # 双向LSTM层
        self.lstm = nn.LSTM(
            input_size=embed_dim + pos_embed_dim * 2,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # 注意力层
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.attention = AttentionLayer(lstm_output_dim)

        # 分类层
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        # 正则化层
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(lstm_output_dim)

    def forward(self, x, head_pos, tail_pos):
        # 嵌入层
        word_embed = self.word_embedding(x)
        head_embed = self.head_pos_embedding(head_pos)
        tail_embed = self.tail_pos_embedding(tail_pos)

        full_embed = torch.cat([word_embed, head_embed, tail_embed], dim=-1)
        full_embed = self.dropout(full_embed)

        # LSTM层
        lstm_output, _ = self.lstm(full_embed)
        lstm_output = self.layer_norm(lstm_output)

        # 注意力层
        context, attn_weights = self.attention(lstm_output)
        context = self.dropout(context)

        # 分类
        logits = self.classifier(context)

        return logits, attn_weights


def evaluate_model(model, data_loader, criterion, device):
    """模型评估函数"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            x, head_pos, tail_pos, labels = batch
            x, head_pos, tail_pos, labels = (
                x.to(device), head_pos.to(device),
                tail_pos.to(device), labels.to(device)
            )

            logits, _ = model(x, head_pos, tail_pos)
            loss = criterion(logits, labels)
            preds = torch.argmax(logits, dim=1)

            total_loss += loss.item() * x.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(data_loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_per_class = f1_score(all_labels, all_preds, average=None)

    return avg_loss, acc, f1_macro, f1_per_class


def train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        device, epochs=30, save_path='att_blstm_best.pth'
):
    """模型训练函数（带早停机制）"""
    model.to(device)
    best_val_f1 = 0.0
    patience = 5
    no_improve_epochs = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            x, head_pos, tail_pos, labels = batch
            x, head_pos, tail_pos, labels = (
                x.to(device), head_pos.to(device),
                tail_pos.to(device), labels.to(device)
            )

            optimizer.zero_grad()
            logits, _ = model(x, head_pos, tail_pos)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)

        avg_train_loss = train_loss / len(train_loader.dataset)
        val_loss, val_acc, val_f1, _ = evaluate_model(model, val_loader, criterion, device)

        print(f"Epoch [{epoch + 1}/{epochs}]")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Val Acc: {val_acc:.4f} | Val F1 (macro): {val_f1:.4f}")
        print("-" * 50)

        scheduler.step(val_loss)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), save_path)
            print(f"最佳模型已保存（Val F1: {best_val_f1:.4f}）")
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print(f"早停于第{epoch + 1}轮（连续{patience}轮无提升）")
                break

    return best_val_f1


def predict_relation(
        model, context_tokens, e1_s, e1_e, e2_s, e2_e,
        word_to_idx, max_seq_len, max_dist, device
):
    """关系预测函数"""
    model.eval()
    relation_map = {0: "无关系", 1: "属于", 2: "创作"}

    # 样本编码
    x = [word_to_idx.get(token, 1) for token in context_tokens]
    if len(x) < max_seq_len:
        x += [0] * (max_seq_len - len(x))
    else:
        x = x[:max_seq_len]
        e1_s, e1_e = min(e1_s, max_seq_len - 1), min(e1_e, max_seq_len - 1)
        e2_s, e2_e = min(e2_s, max_seq_len - 1), min(e2_e, max_seq_len - 1)

    # 计算相对位置
    head_center = (e1_s + e1_e) // 2
    tail_center = (e2_s + e2_e) // 2
    head_pos = []
    tail_pos = []
    for i in range(max_seq_len):
        d_h = min(max(i - head_center, -max_dist), max_dist) + max_dist
        d_t = min(max(i - tail_center, -max_dist), max_dist) + max_dist
        head_pos.append(d_h)
        tail_pos.append(d_t)

    # 转换为张量
    x = torch.tensor(x, dtype=torch.long).unsqueeze(0).to(device)
    head_pos = torch.tensor(head_pos, dtype=torch.long).unsqueeze(0).to(device)
    tail_pos = torch.tensor(tail_pos, dtype=torch.long).unsqueeze(0).to(device)

    # 预测
    with torch.no_grad():
        logits, attn_weights = model(x, head_pos, tail_pos)
        pred_idx = torch.argmax(logits, dim=1).item()

    return relation_map[pred_idx], attn_weights.squeeze(0).cpu().numpy()


def main():
    # 配置参数
    BIO_PATH = "./entityBIO_corrected.txt"
    BATCH_SIZE = 32
    EMBED_DIM = 100
    HIDDEN_DIM = 128
    NUM_LAYERS = 2
    DROPOUT = 0.5
    LEARNING_RATE = 1e-3
    EPOCHS = 30
    SAVE_PATH = "att_blstm_relation_best.pth"
    MAX_DIST = 50

    # 数据预处理（获取分割后的样本列表）
    print("数据预处理中...")
    (train_dataset, val_dataset, test_dataset,
     word_to_idx, vocab_size, max_seq_len, max_dist,
     train_samples, val_samples, test_samples) = preprocess_bio_data(  # 新增返回的样本列表
        BIO_PATH, MAX_DIST
    )

    # 构建数据加载器
    train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"数据集信息：")
    print(f"训练集: {len(train_dataset)} | 验证集: {len(val_dataset)} | 测试集: {len(test_dataset)}")
    print(f"词表大小: {vocab_size} | 最大序列长度: {max_seq_len}")

    # 模型初始化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttBLSTMRelationExtractor(
        vocab_size=vocab_size,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_classes=3,
        max_dist=MAX_DIST,
        dropout=DROPOUT
    )

    # 损失函数与优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )

    # 学习率衰减
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-6
    )

    # 训练模型
    print("\n开始训练Att-BLSTM模型...")
    best_val_f1 = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        device, EPOCHS, SAVE_PATH
    )

    # 测试模型
    print("\n测试最佳模型...")
    model.load_state_dict(torch.load(SAVE_PATH))
    test_loss, test_acc, test_f1, test_f1_per_class = evaluate_model(model, test_loader, criterion, device)
    print(f"测试集结果：")
    print(f"Loss: {test_loss:.4f} | Acc: {test_acc:.4f} | F1(macro): {test_f1:.4f}")
    print(
        f"每类F1值: 无关系={test_f1_per_class[0]:.4f}, 属于={test_f1_per_class[1]:.4f}, 创作={test_f1_per_class[2]:.4f}")

    # 预测示例（修复部分）
    if len(test_samples) > 0:  # 直接使用分割后的测试样本列表
        print("\n预测示例：")
        test_sample_idx = 0
        raw_sample = test_samples[test_sample_idx]  # 直接从测试样本列表获取
        context_tokens, e1_s, e1_e, e2_s, e2_e, true_rel = raw_sample
        e1_text = ''.join(context_tokens[e1_s:e1_e + 1])
        e2_text = ''.join(context_tokens[e2_s:e2_e + 1])

        pred_rel, attn_weights = predict_relation(
            model, context_tokens, e1_s, e1_e, e2_s, e2_e,
            word_to_idx, max_seq_len, max_dist, device
        )

        true_rel_text = {0: "无关系", 1: "属于", 2: "创作"}[true_rel]
        print(f"实体对: {e1_text} ↔ {e2_text}")
        print(f"真实关系: {true_rel_text} | 预测关系: {pred_rel}")
        print(f"注意力权重前5位: {attn_weights[:5].round(4)}")


if __name__ == "__main__":
    main()
