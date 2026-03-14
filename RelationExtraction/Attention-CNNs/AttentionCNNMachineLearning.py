import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torch.nn import functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from collections import Counter
from gensim.models import KeyedVectors
from torch.nn.utils.parametrizations import weight_norm  # 需要导入这个工具函数
"""
优化说明：
1. 增加预训练词嵌入支持（同CR-CNN）
2. 增强正则化（L2正则、更高Dropout、早停机制）
3. 优化位置嵌入（采用相对距离编码，同CR-CNN）
4. 降低模型容量（减少卷积核和隐藏层维度）
5. 加入学习率衰减策略
6. 统一数据处理流程与CR-CNN对齐
"""


def preprocess_bio_data(bio_path, embedding_path):
    """
    解析BIO标注文件，提取实体并构建关系样本（增加预训练词嵌入支持）
    :param bio_path: BIO标注文件路径
    :param embedding_path: 预训练词嵌入路径
    :return: 数据集、词表、嵌入矩阵、最大序列长度
    """
    # 1. 读取BIO文件
    bio_data = []
    with open(bio_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            token, label = line.split()
            bio_data.append((token, label))

    # 2. 分割文本片段
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

    # 3. 构建关系样本
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

    # 4. 构建词表并加载预训练嵌入
    word_counts = Counter()
    for sample in samples:
        word_counts.update(sample[0])

    # 加载预训练词向量
    w2v_model = KeyedVectors.load_word2vec_format(embedding_path, binary=False)
    embed_dim = w2v_model.vector_size

    # 构建词表（确保预训练词在词表中）
    word_to_idx = {'<PAD>': 0, '<UNK>': 1}
    for word, _ in word_counts.most_common():
        if word not in word_to_idx:
            word_to_idx[word] = len(word_to_idx)

    # 补充预训练词表中存在但语料中未出现的高频词
    for word in w2v_model.index_to_key[:10000]:  # 取前10000个高频词
        if word not in word_to_idx:
            word_to_idx[word] = len(word_to_idx)

    vocab_size = len(word_to_idx)

    # 构建嵌入矩阵
    embedding_matrix = np.zeros((vocab_size, embed_dim))
    for word, idx in word_to_idx.items():
        if word in w2v_model:
            embedding_matrix[idx] = w2v_model[word]
        else:
            embedding_matrix[idx] = np.random.normal(0, 0.1, embed_dim)

    # 5. 确定最大序列长度
    max_seq_len = max(len(sample[0]) for sample in samples) if samples else 128

    # 6. 样本编码（增加相对位置编码）
    def encode_sample(sample, max_dist=50):
        context_tokens, e1_s, e1_e, e2_s, e2_e, relation = sample
        seq_len = len(context_tokens)

        # 词编码
        x = [word_to_idx.get(token, 1) for token in context_tokens]
        if len(x) < max_seq_len:
            x += [0] * (max_seq_len - len(x))
        else:
            x = x[:max_seq_len]
            e1_e = min(e1_e, max_seq_len - 1)
            e2_e = min(e2_e, max_seq_len - 1)
            e1_s = min(e1_s, max_seq_len - 1)
            e2_s = min(e2_s, max_seq_len - 1)

        # 计算实体中心
        head_center = (e1_s + e1_e) // 2
        tail_center = (e2_s + e2_e) // 2

        # 相对位置编码（同CR-CNN）
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

    # 7. 划分数据集
    train_samples, val_test_samples = train_test_split(samples, test_size=0.3, random_state=42)
    val_samples, test_samples = train_test_split(val_test_samples, test_size=0.5, random_state=42)

    # 8. 构建数据集类
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
        word_to_idx, embedding_matrix, vocab_size, max_seq_len, embed_dim
    )


class AttentionCNNRelationExtractor(nn.Module):
    def __init__(
            self, vocab_size, embedding_matrix, embed_dim, max_dist=50,
            num_filters=64, kernel_sizes=[3, 5, 7], hidden_dim=128,
            num_classes=3, dropout=0.6
    ):
        """优化后的模型：降低容量+增强正则化"""
        super().__init__()

        # 1. 词嵌入层（使用预训练权重）
        self.word_embedding = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix, dtype=torch.float32),
            padding_idx=0,
            freeze=False  # 允许微调
        )

        # 2. 位置嵌入层（相对距离）
        pos_embed_dim = embed_dim // 2
        self.head_pos_embedding = nn.Embedding(
            2 * max_dist + 1, pos_embed_dim, padding_idx=0
        )
        self.tail_pos_embedding = nn.Embedding(
            2 * max_dist + 1, pos_embed_dim, padding_idx=0
        )

        # 3. 卷积层（添加权重正则化）
        self.convs = nn.ModuleList([
            weight_norm(  # 使用weight_norm函数包装卷积层
                nn.Conv2d(
                    in_channels=1,
                    out_channels=num_filters,
                    kernel_size=(k, embed_dim + pos_embed_dim * 2),
                    padding=(k // 2, 0)
                )
            ) for k in kernel_sizes
        ])

        # 4. 自注意力层（降低复杂度）
        self.attention = nn.MultiheadAttention(
            embed_dim=num_filters,
            num_heads=2,  # 多头注意力但减少头数
            dropout=dropout,
            batch_first=True
        )

        # 5. 分类层（缩减维度+正则化）
        self.classifier = nn.Sequential(
            nn.Linear(num_filters * len(kernel_sizes), hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

        # 6. 正则化层
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(num_filters)  # 层归一化稳定训练

    def forward(self, x, head_pos, tail_pos):
        # 1. 嵌入层
        word_embed = self.word_embedding(x)  # [batch, seq_len, embed_dim]
        head_embed = self.head_pos_embedding(head_pos)  # [batch, seq_len, pos_dim]
        tail_embed = self.tail_pos_embedding(tail_pos)  # [batch, seq_len, pos_dim]

        # 拼接嵌入特征
        full_embed = torch.cat([word_embed, head_embed, tail_embed], dim=-1)  # [batch, seq_len, total_dim]
        full_embed = self.dropout(full_embed)
        full_embed = full_embed.unsqueeze(1)  # [batch, 1, seq_len, total_dim]

        # 2. 卷积提取特征
        conv_features = []
        for conv in self.convs:
            conv_out = conv(full_embed)  # [batch, num_filters, seq_len, 1]
            conv_out = conv_out.squeeze(-1).transpose(1, 2)  # [batch, seq_len, num_filters]
            conv_out = F.relu(conv_out)
            conv_features.append(conv_out)

        # 3. 注意力增强
        attn_features = []
        for feat in conv_features:
            attn_out, _ = self.attention(feat, feat, feat)  # [batch, seq_len, num_filters]
            attn_out = self.layer_norm(attn_out + feat)  # 残差连接+层归一化
            pool_out = torch.max(attn_out, dim=1)[0]  # [batch, num_filters]
            attn_features.append(pool_out)

        # 4. 分类
        concat_feat = torch.cat(attn_features, dim=1)  # [batch, num_filters*len(kernels)]
        logits = self.classifier(concat_feat)

        return logits


def evaluate_model(model, data_loader, criterion, device):
    """优化评估函数：增加每类F1计算"""
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

            logits = model(x, head_pos, tail_pos)
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
        device, epochs=30, save_path='attention_cnn_best.pth'
):
    """优化训练函数：增加早停和学习率衰减"""
    model.to(device)
    best_val_f1 = 0.0
    patience = 5  # 早停耐心值
    no_improve_epochs = 0

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            x, head_pos, tail_pos, labels = batch
            x, head_pos, tail_pos, labels = (
                x.to(device), head_pos.to(device),
                tail_pos.to(device), labels.to(device)
            )

            optimizer.zero_grad()
            logits = model(x, head_pos, tail_pos)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)

        avg_train_loss = train_loss / len(train_loader.dataset)

        # 验证阶段
        val_loss, val_acc, val_f1, _ = evaluate_model(model, val_loader, criterion, device)

        # 打印信息
        print(f"Epoch [{epoch + 1}/{epochs}]")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Val Acc: {val_acc:.4f} | Val F1 (macro): {val_f1:.4f}")
        print("-" * 50)

        # 学习率衰减
        scheduler.step(val_loss)

        # 早停机制
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved (Val F1: {best_val_f1:.4f})")
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print(f"早停于第{epoch + 1}轮（{patience}轮无提升）")
                break

    return best_val_f1


def predict_relation(
        model, context_tokens, e1_s, e1_e, e2_s, e2_e,
        word_to_idx, max_seq_len, device, max_dist=50
):
    """优化预测函数：适配新的位置编码"""
    model.eval()
    relation_map = {0: "无关系", 1: "属于", 2: "创作"}

    # 编码
    x = [word_to_idx.get(token, 1) for token in context_tokens]
    if len(x) < max_seq_len:
        x += [0] * (max_seq_len - len(x))
    else:
        x = x[:max_seq_len]
        e1_e = min(e1_e, max_seq_len - 1)
        e2_e = min(e2_e, max_seq_len - 1)
        e1_s = min(e1_s, max_seq_len - 1)
        e2_s = min(e2_s, max_seq_len - 1)

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
        logits = model(x, head_pos, tail_pos)
        pred_idx = torch.argmax(logits, dim=1).item()

    return relation_map[pred_idx]


def main():
    # 配置参数
    BIO_PATH = "./entityBIO_corrected.txt"
    EMBEDDING_PATH = r"../CR-CNN/sgns.target.word-character.char1-2.dynwin5.thr10.neg5.dim300.iter5"  # 替换为实际预训练词向量路径
    BATCH_SIZE = 32
    NUM_FILTERS = 64  # 减少卷积核数量
    KERNEL_SIZES = [3, 5, 7]
    HIDDEN_DIM = 128  # 缩减隐藏层维度
    DROPOUT = 0.6  # 提高Dropout比例
    LEARNING_RATE = 1e-3
    EPOCHS = 30
    SAVE_PATH = "attention_cnn_optimized.pth"
    MAX_DIST = 50  # 位置编码最大距离

    # 数据预处理
    print("数据预处理中...")
    train_dataset, val_dataset, test_dataset, word_to_idx, embedding_matrix, vocab_size, max_seq_len, embed_dim = preprocess_bio_data(
        BIO_PATH, EMBEDDING_PATH
    )

    # 数据加载器
    train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"数据集信息：")
    print(f"训练集: {len(train_dataset)} | 验证集: {len(val_dataset)} | 测试集: {len(test_dataset)}")
    print(f"词表大小: {vocab_size} | 最大序列长度: {max_seq_len}")

    # 模型初始化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttentionCNNRelationExtractor(
        vocab_size=vocab_size,
        embedding_matrix=embedding_matrix,
        embed_dim=embed_dim,
        max_dist=MAX_DIST,
        num_filters=NUM_FILTERS,
        kernel_sizes=KERNEL_SIZES,
        hidden_dim=HIDDEN_DIM,
        num_classes=3,
        dropout=DROPOUT
    )

    # 损失函数与优化器（添加权重衰减）
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4  # L2正则化
    )

    # 学习率衰减
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-6
    )

    # 训练模型
    print("\n开始训练...")
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
    print(f"每类F1: {test_f1_per_class}")

    # 预测示例
    if len(test_dataset) > 0:
        print("\n预测示例：")
        # 设置要显示的示例数量，可以根据需要调整
        num_examples = 5  # 例如，显示5个预测示例
        # 确保示例数量不超过测试数据集的大小
        num_examples = min(num_examples, len(test_dataset.samples))
        
        for i in range(num_examples):
            raw_sample = test_dataset.samples[i]
            context_tokens, e1_s, e1_e, e2_s, e2_e, true_rel = raw_sample
            e1_text = ''.join(context_tokens[e1_s:e1_e + 1])
            e2_text = ''.join(context_tokens[e2_s:e2_e + 1])

            pred_rel = predict_relation(
                model, context_tokens, e1_s, e1_e, e2_s, e2_e,
                word_to_idx, max_seq_len, device
            )

            true_rel_text = {0: "无关系", 1: "属于", 2: "创作"}[true_rel]
            print(f"\n示例 {i+1}：")
            print(f"实体对：{e1_text} ↔ {e2_text}")
            print(f"真实关系：{true_rel_text} | 预测关系：{pred_rel}")



if __name__ == "__main__":
    main()