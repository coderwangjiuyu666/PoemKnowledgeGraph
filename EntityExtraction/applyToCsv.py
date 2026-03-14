import torch
import pandas as pd
from collections import defaultdict
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data import Dataset, DataLoader
from torchcrf import CRF

vocab = {"<PAD>": 0, "<UNK>": 1}  # 示例词汇表
# 构建标签表
all_tags = set()
all_tags.add("<PAD>")  # 新增Padding标签
tag_to_id = {tag: idx for idx, tag in enumerate(all_tags)}
id_to_tag = {idx: tag for tag, idx in tag_to_id.items()}
PAD_ID = tag_to_id["<PAD>"]  # Padding标签的ID


# 加载训练好的模型
class BiLSTM_CRF(nn.Module):  # 复用模型定义
    def __init__(self, vocab_size, tag_size, embedding_dim=128, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.bilstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, tag_size)
        self.crf = CRF(tag_size, batch_first=True)

    def forward(self, x, pos_x=None, tags=None, mask=None):
        embed = self.embedding(x)
        lstm_out, _ = self.bilstm(embed)
        logits = self.fc(lstm_out)
        if tags is not None:
            return -self.crf(logits, tags, mask=mask)
        else:
            return self.crf.decode(logits, mask=mask)


# 初始化模型并加载权重
vocab_size = len(vocab)
tag_size = len(tag_to_id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BiLSTM_CRF(vocab_size, tag_size)
model.load_state_dict(torch.load("best_bilstm_crf.pth", map_location=device))
model.to(device)
model.eval()


# 文本预处理函数
def preprocess_text(text):
    """将文本转换为模型输入格式"""
    # 按字符分割（中文通常按单字处理）
    chars = list(text)
    # 转换为ID
    text_ids = [vocab.get(char, vocab["<UNK>"]) for char in chars]
    # 生成位置特征（与训练时保持一致）
    pos_x = []
    for i, char in enumerate(chars):
        is_start = 1 if i == 0 else 0
        is_punc = 1 if char in [";", "，", "。", "、", ".", "!", "?"] else 0
        pos_x.append(is_start * 2 + is_punc)
    return torch.tensor([text_ids]), torch.tensor([pos_x]), chars


# 实体提取函数（从BIO标签中解析实体）
def extract_entities(chars, tags):
    """从字符列表和标签列表中提取实体"""
    entities = []
    current_entity = None
    current_type = None

    for char, tag in zip(chars, tags):
        if tag.startswith("B-"):
            # 新实体开始
            if current_entity:
                entities.append({
                    "实体文本": current_entity,
                    "实体类型": current_type
                })
            current_type = tag[2:]  # 提取B-后的类型
            current_entity = char
        elif tag.startswith("I-") and current_entity:
            # 实体延续
            current_entity += char
        else:
            # 非实体或实体结束
            if current_entity:
                entities.append({
                    "实体文本": current_entity,
                    "实体类型": current_type
                })
                current_entity = None
                current_type = None
    # 处理最后一个实体
    if current_entity:
        entities.append({
            "实体文本": current_entity,
            "实体类型": current_type
        })
    return entities


# 批量处理文本并导出到CSV
def process_and_export(texts, output_path="诗文作者朝代实体抽取结果.csv"):
    """处理文本列表，提取实体并导出到CSV"""
    all_results = []

    for idx, text in enumerate(texts):
        # 预处理
        text_ids, pos_x, chars = preprocess_text(text)
        text_ids = text_ids.to(device)
        pos_x = pos_x.to(device)
        mask = (text_ids != vocab["<PAD>"]).to(device)  # 生成mask

        # 模型预测
        with torch.no_grad():
            pred_tag_ids = model(text_ids, pos_x, mask=mask)[0]  # 取第一个样本的预测结果

        # 转换为标签文本
        pred_tags = [id_to_tag[tag_id] for tag_id in pred_tag_ids]

        # 提取实体
        entities = extract_entities(chars, pred_tags)

        # 补充文本信息
        for entity in entities:
            all_results.append({
                "文本ID": idx,
                "原始文本": text,
                **entity
            })

    # 导出到CSV
    df = pd.DataFrame(all_results)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"实体抽取完成，结果已保存至 {output_path}")

def load_texts_from_txt(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]
    return texts
# 示例使用
if __name__ == "__main__":
    test_texts = load_texts_from_txt('input_texts.txt')
    process_and_export(test_texts)