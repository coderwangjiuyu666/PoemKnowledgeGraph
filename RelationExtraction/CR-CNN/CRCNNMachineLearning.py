import os
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, precision_score, recall_score
import tensorflow as tf
import keras
from keras import layers, models, optimizers, losses, callbacks,regularizers
from keras.preprocessing.sequence import pad_sequences
from gensim.models import KeyedVectors  # 用于加载预训练Word2Vec

# 步骤 1：解析 BIO 文件，提取实体与文本块
def parse_bio_by_entity_matching(bio_path):
    """
    修正：修复缩进错误、调整校验时机、增加调试打印
    """
    # 第一步：全量提取所有实体（代码不变）
    all_entities = []
    current_entity = None
    token_idx = 0

    with open(bio_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
        total_lines = len(lines)

        for line_idx, line in enumerate(lines):
            token, label = line.split()

            if label.startswith("B-"):
                if current_entity is not None:
                    all_entities.append(current_entity)
                ent_type = label.split("-")[1]
                current_entity = {
                    "type": ent_type,
                    "content": token,
                    "start_line": line_idx,
                    "end_line": line_idx,
                    "start_token": token_idx,
                    "end_token": token_idx
                }

            elif label.startswith("I-"):
                if current_entity is None or current_entity["type"] != label.split("-")[1]:
                    continue
                current_entity["content"] += token
                current_entity["end_line"] = line_idx
                current_entity["end_token"] = token_idx

            token_idx += 1

        if current_entity is not None:
            all_entities.append(current_entity)

    # 打印提取的所有实体（调试用）
    print("\n【提取的所有实体】")
    for i, ent in enumerate(all_entities):
        print(f"实体{i}：类型={ent['type']}，内容={ent['content']}，行范围=[{ent['start_line']}-{ent['end_line']}]")

    # 第二步：按“Poem→Author→Dynasty”配对，生成诗歌块
    poems = []
    used_poem_indices = set()

    for auth_idx, auth in enumerate(all_entities):
        if auth["type"] != "Author":
            continue

        # 1. 找对应的Dynasty
        dyn = None
        for dyn_idx in range(auth_idx + 1, len(all_entities)):
            candidate = all_entities[dyn_idx]
            if candidate["type"] == "Dynasty":
                dyn = candidate
                break
            elif candidate["type"] == "Author":
                break

        # 2. 找对应的Poem
        poem = None
        for poem_idx in range(auth_idx - 1, -1, -1):
            candidate = all_entities[poem_idx]
            if candidate["type"] == "Poem" and poem_idx not in used_poem_indices:
                poem = candidate
                used_poem_indices.add(poem_idx)
                break

        # 调试打印：当前Author的配对结果
        print(f"\n【处理Author】{auth['content']}（行号：{auth['start_line']}）")
        print(f"  找到Dynasty：{dyn['content'] if dyn else '无'}")
        print(f"  找到Poem：{poem['content'] if poem else '无'}")

        # 若Dynasty和Poem都找到，才生成诗歌样本（修复缩进：这部分必须在if外执行）
        if dyn is not None and poem is not None:
            # 3. 提取当前诗的文本token（局部序列）
            poem_start_line = poem["start_line"]
            dyn_end_line = dyn["end_line"]
            poem_text = []
            local_to_global_token = []
            for line_idx in range(poem_start_line, dyn_end_line + 1):
                token = lines[line_idx].split()[0]
                poem_text.append(token)
                # 全局token_idx = 行索引（因已过滤空行，与token_idx累计一致）
                local_to_global_token.append(line_idx)

            # 4. 计算实体在局部文本中的位置
            def get_local_pos(global_start, global_end, local_to_global):
                try:
                    local_start = local_to_global.index(global_start)
                    local_end = len(local_to_global) - 1 - local_to_global[::-1].index(global_end)
                    return {"start": local_start, "end": local_end}
                except ValueError:
                    print(f"  警告：实体全局位置[{global_start}-{global_end}]不在局部文本中")
                    return {"start": 0, "end": 0}

            poem_local_pos = get_local_pos(poem["start_token"], poem["end_token"], local_to_global_token)
            auth_local_pos = get_local_pos(auth["start_token"], auth["end_token"], local_to_global_token)
            dyn_local_pos = get_local_pos(dyn["start_token"], dyn["end_token"], local_to_global_token)

            # 更新实体局部位置
            poem["local_start"], poem["local_end"] = poem_local_pos["start"], poem_local_pos["end"]
            auth["local_start"], auth["local_end"] = auth_local_pos["start"], auth_local_pos["end"]
            dyn["local_start"], dyn["local_end"] = dyn_local_pos["start"], dyn_local_pos["end"]

            # 添加到诗歌列表
            poems.append({
                "text": poem_text,
                "text_length": len(poem_text),
                "entities": [poem, auth, dyn],
                "global_token_range": [poem["start_token"], dyn["end_token"]],
                "local_token_range": [0, len(poem_text) - 1]
            })
            print(f"  生成诗歌样本：文本长度={len(poem_text)}，实体数={len(poems[-1]['entities'])}")

    # 关键修正：所有Author处理完毕后，再进行实体完整性校验（移到循环外）
    valid_poems = []
    for poem in poems:
        ent_types = [ent["type"] for ent in poem["entities"]]
        if ent_types.count("Poem") == 1 and ent_types.count("Author") == 1 and ent_types.count("Dynasty") == 1:
            valid_poems.append(poem)
        else:
            print(f"过滤无效样本：实体类型{ent_types}")
    poems = valid_poems
    print(f"\n【最终有效诗歌样本数】：{len(poems)}")

    return poems, lines

# ------------------- 执行解析 -------------------
bio_path = r"./entityBIO_corrected.txt"
poems, original_lines = parse_bio_by_entity_matching(bio_path)

# ------------------- 验证结果 -------------------
# print(f"最终解析出 {len(poems)} 首完整诗歌\n")
# # 打印前3首，确保每首只有1个Poem、1个Author、1个Dynasty
# for i in range(min(3, len(poems))):
#     poem = poems[i]
#     print(f"【第{i+1}首诗】")
#     print(f"文本长度：{poem['text_length']} 个token")
#     print(f"完整文本：{''.join(poem['text'])}")
#     print("实体信息：")
#     for ent in poem["entities"]:
#         print(f"  - 类型：{ent['type']:6s} | 内容：{ent['content']:40s} | 行范围：[{ent['start_line']}-{ent['end_line']}] | token范围：[{ent['start_token']}-{ent['end_token']}]")
#     print("=" * 120)

# 步骤 2：生成实体对并自动标注关系
def generate_labeled_pairs(text_blocks):
    label_map = {0: "无关系", 1: "属于", 2: "创作"}
    pairs = []

    for block in text_blocks:
        text_seq = block["text"]
        entities = block["entities"]
        entity_dict = defaultdict(list)
        for ent in entities:
            entity_dict[ent["type"]].append(ent)

        # 必须包含Author、Poem、Dynasty才生成样本
        if not (entity_dict.get("Author") and entity_dict.get("Poem") and entity_dict.get("Dynasty")):
            continue

        author = entity_dict["Author"][0]
        poem = entity_dict["Poem"][0]
        dynasty = entity_dict["Dynasty"][0]

        # 1. 正样本：属于（Author→Dynasty）、创作（Author→Poem）
        pairs.append({"text_seq": text_seq, "head": author, "tail": dynasty, "label": 1})
        pairs.append({"text_seq": text_seq, "head": author, "tail": poem, "label": 2})

        # 2. 负样本：仅保留2类（而非4类）→ 每首诗生成2个负样本
        pairs.append({"text_seq": text_seq, "head": poem, "tail": dynasty, "label": 0})  # Poem→Dynasty
        pairs.append({"text_seq": text_seq, "head": dynasty, "tail": poem, "label": 0})  # Dynasty→Poem

    print(f"生成样本数：{len(pairs)}，类别分布：{np.bincount([p['label'] for p in pairs])}")
    return pairs, label_map

# ------------------- 调用函数生成标注样本 -------------------
labeled_pairs, label_map = generate_labeled_pairs(poems)
# print(f"生成 {len(labeled_pairs)} 个实体对样本")
# # 示例：打印前3个样本的标签和实体对
# for i in range(3):
#     print(
#         f"样本{i + 1}：{labeled_pairs[i]['head']['content']} → {labeled_pairs[i]['tail']['content']}，标签：{label_map[labeled_pairs[i]['label']]}")


# 步骤 3：特征工程（词嵌入 + 位置嵌入）
# 加载预训练词嵌入
def build_word_embedding(vocab, embedding_path, embedding_dim=300):
    """
    构建词嵌入矩阵
    :param vocab: 词汇表（所有文本的token集合）
    :param embedding_path: 预训练Word2Vec路径
    :param embedding_dim: 嵌入维度
    :return: 词嵌入矩阵（shape: (vocab_size, embedding_dim)）
    """
    # 加载预训练Word2Vec
    w2v_model = KeyedVectors.load_word2vec_format(embedding_path, binary=False)

    # 初始化嵌入矩阵（未知词用随机向量，PAD用全零）
    vocab_size = len(vocab)
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    word2idx = {word: idx for idx, word in enumerate(vocab)}

    for word, idx in word2idx.items():
        if word in w2v_model:
            embedding_matrix[idx] = w2v_model[word]
        else:
            # 未知词：随机初始化（均值0，标准差0.1）
            embedding_matrix[idx] = np.random.normal(0, 0.1, embedding_dim)

    return embedding_matrix, word2idx


# ------------------- 构建词汇表和词嵌入 -------------------
# 1. 收集所有文本的token，构建词汇表
all_tokens = [token for pair in labeled_pairs for token in pair["text_seq"]]
vocab = list(set(all_tokens)) + ["[PAD]"]  # 加入PAD符号
vocab_size = len(vocab)
word2idx = {word: idx for idx, word in enumerate(vocab)}
print(f"词汇表大小：{vocab_size}")
# 2. 加载预训练Word2Vec并构建嵌入矩阵（替换为你的Word2Vec路径）
w2v_path = r"./sgns.target.word-character.char1-2.dynwin5.thr10.neg5.dim300.iter5"
embedding_matrix, _ = build_word_embedding(vocab, w2v_path, embedding_dim=300)

# 计算位置嵌入与特征拼接
def compute_position_embedding(seq_len, head_pos, tail_pos, pos_dim=50, max_dist=50):
    """
    计算位置嵌入：每个token到head和tail的相对距离
    :param seq_len: 文本序列长度
    :param head_pos: 头实体的{start, end}
    :param tail_pos: 尾实体的{start, end}
    :param pos_dim: 位置嵌入维度
    :param max_dist: 最大距离（超过则截断）
    :return: 位置嵌入矩阵（shape: (seq_len, 2*pos_dim)）
    """
    # 计算实体中心位置
    head_center = (head_pos["start"] + head_pos["end"]) // 2
    tail_center = (tail_pos["start"] + tail_pos["end"]) // 2

    # 初始化位置嵌入表（随机初始化，后续训练优化）
    pos_embedding_table = np.random.normal(0, 0.1, (2 * max_dist + 1, pos_dim))

    # 计算每个token的位置嵌入
    pos_emb = []
    for i in range(seq_len):
        # 计算到head和tail的距离，截断到[-max_dist, max_dist]
        d_head = min(max(i - head_center, -max_dist), max_dist) + max_dist  # 映射到[0, 2*max_dist]
        d_tail = min(max(i - tail_center, -max_dist), max_dist) + max_dist

        # 拼接head和tail的位置嵌入
        emb_h = pos_embedding_table[d_head]
        emb_t = pos_embedding_table[d_tail]
        pos_emb.append(np.concatenate([emb_h, emb_t]))

    return np.array(pos_emb)


def build_features(labeled_pairs, word2idx, max_seq_len=128, max_dist=50):
    """输出3个特征：词索引、head位置索引、tail位置索引"""
    X_word = []  # 词索引序列 (n_samples, max_seq_len)
    X_head_pos = []  # head实体相对距离索引 (n_samples, max_seq_len)
    X_tail_pos = []  # tail实体相对距离索引 (n_samples, max_seq_len)
    y = []

    for pair in labeled_pairs:
        text_seq = pair["text_seq"]
        # 关键修改：使用局部位置（local_start/local_end）
        head = {"start": pair["head"]["local_start"], "end": pair["head"]["local_end"]}
        tail = {"start": pair["tail"]["local_start"], "end": pair["tail"]["local_end"]}
        label = pair["label"]

        # 1. 词序列转索引（PAD/截断）
        seq_idx = [word2idx.get(word, word2idx["[PAD]"]) for word in text_seq]
        seq_idx = pad_sequences([seq_idx], maxlen=max_seq_len, padding="post", truncating="post")[0]
        X_word.append(seq_idx)

        # 2. 计算实体相对距离（映射为索引：[-max_dist, max_dist] → [0, 2*max_dist]）
        head_center = (head["start"] + head["end"]) // 2  # 实体中心位置
        tail_center = (tail["start"] + tail["end"]) // 2
        head_dist = []
        tail_dist = []

        for i in range(max_seq_len):
            # 截断距离（避免极端值）
            d_h = min(max(i - head_center, -max_dist), max_dist) + max_dist
            d_t = min(max(i - tail_center, -max_dist), max_dist) + max_dist
            head_dist.append(d_h)
            tail_dist.append(d_t)

        X_head_pos.append(head_dist)
        X_tail_pos.append(tail_dist)

        # 3. 标签One-Hot编码
        y.append(tf.keras.utils.to_categorical(label, num_classes=3))

    return (np.array(X_word), np.array(X_head_pos), np.array(X_tail_pos), np.array(y))

# ------------------- 生成模型输入特征 -------------------
# 1. 生成特征（调用重构后的build_features）
max_seq_len = 128
max_dist = 50  # 位置距离最大阈值
X_word, X_head_pos, X_tail_pos, y = build_features(labeled_pairs, word2idx, max_seq_len, max_dist)

# 2. 生成样本索引（用于后续匹配测试集实体对）
indices = np.arange(len(X_word))

# 3. 分层抽样划分数据集（7:2:1）
# 第一次划分：训练集70%，临时集30%
train_indices, temp_indices = train_test_split(
    indices, test_size=0.3, random_state=42, stratify=y  # stratify：按类别分布抽样
)
# 第二次划分：验证集20%，测试集10%
val_indices, test_indices = train_test_split(
    temp_indices, test_size=1/3, random_state=42, stratify=y[temp_indices]
)

# 4. 按索引提取训练/验证/测试集
# 训练集
X_word_train, X_head_pos_train, X_tail_pos_train = X_word[train_indices], X_head_pos[train_indices], X_tail_pos[train_indices]
y_train = y[train_indices]
# 验证集
X_word_val, X_head_pos_val, X_tail_pos_val = X_word[val_indices], X_head_pos[val_indices], X_tail_pos[val_indices]
y_val = y[val_indices]
# 测试集
X_word_test, X_head_pos_test, X_tail_pos_test = X_word[test_indices], X_head_pos[test_indices], X_tail_pos[test_indices]
y_test = y[test_indices]

# 5. 匹配测试集对应的实体对（用于推理）
test_pairs = [labeled_pairs[i] for i in test_indices]

print(f"训练集：{X_word_train.shape}，验证集：{X_word_val.shape}，测试集：{X_word_test.shape}")


# 步骤 4：搭建 CR-CNN 模型
def build_cr_cnn_model(vocab_size, embedding_matrix, max_seq_len, pos_dim=50, num_classes=3, max_dist=50):
    """多输入模型：词输入 + head位置输入 + tail位置输入"""
    # 1. 输入层（三个独立输入）
    input_word = layers.Input(shape=(max_seq_len,), name="input_word")
    input_head_pos = layers.Input(shape=(max_seq_len,), name="input_head_pos")
    input_tail_pos = layers.Input(shape=(max_seq_len,), name="input_tail_pos")

    # 2. 词嵌入层（使用预训练权重，可微调）
    word_emb = layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_matrix.shape[1],
        weights=[embedding_matrix],
        input_length=max_seq_len,
        trainable=True  # 设为True：微调预训练词嵌入（数据量足够时推荐）
    )(input_word)  # shape: (None, 128, 300)

    # 3. 可训练的位置嵌入层（head + tail）
    head_pos_emb = layers.Embedding(
        input_dim=2 * max_dist + 1,  # 距离索引范围：0~100（max_dist=50）
        output_dim=pos_dim,
        input_length=max_seq_len,
        trainable=True
    )(input_head_pos)  # shape: (None, 128, 50)

    tail_pos_emb = layers.Embedding(
        input_dim=2 * max_dist + 1,
        output_dim=pos_dim,
        input_length=max_seq_len,
        trainable=True
    )(input_tail_pos)  # shape: (None, 128, 50)

    # 4. 拼接特征（词嵌入300 + 位置嵌入50*2 = 400维，与原输入维度一致）
    combined_emb = layers.Concatenate(axis=-1)([word_emb, head_pos_emb, tail_pos_emb])

    # 5. 卷积层（添加L2正则化防过拟合）
    filter_sizes = [3, 5, 7]  # 捕捉不同长度的局部特征
    filter_num = 64  # 减少卷积核数量，降低模型复杂度
    conv_outputs = []

    for fs in filter_sizes:
        conv = layers.Conv1D(
            filters=filter_num,
            kernel_size=fs,
            padding="same",
            activation="relu",
            kernel_regularizer=regularizers.l2(1e-4)  # L2正则化抑制过拟合
        )(combined_emb)
        pool = layers.GlobalMaxPooling1D()(conv)  # 全局最大池化
        conv_outputs.append(pool)

    # 6. 特征拼接 + Dropout + 全连接层
    concat = layers.Concatenate()(conv_outputs)  # shape: (None, 64*3=192)
    dropout = layers.Dropout(0.6)(concat)  # 提高Dropout比例，增强泛化
    output = layers.Dense(
        num_classes,
        activation="softmax",
        kernel_regularizer=regularizers.l2(1e-4)  # 全连接层正则化
    )(dropout)

    # 构建多输入模型
    model = models.Model(
        inputs=[input_word, input_head_pos, input_tail_pos],
        outputs=output
    )

    # 编译模型
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss=losses.CategoricalCrossentropy(),
        metrics=["accuracy"]
    )
    return model

# ------------------- 初始化CR-CNN模型 -------------------
model = build_cr_cnn_model(
    vocab_size=vocab_size,
    embedding_matrix=embedding_matrix,
    max_seq_len=128,
    pos_dim=50,
    num_classes=3,
    max_dist=50
)
model.summary()

# 步骤 5：模型训练（含早停防止过拟合）
def train_model(model, X_train_list, y_train, X_val_list, y_val, epochs=30, batch_size=32):
    # 1. 早停回调：验证损失3轮不下降则停止，恢复最优权重
    early_stopping = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1
    )

    # 2. 学习率衰减：验证损失2轮不下降则减半学习率
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-5,
        verbose=1
    )

    # 3. 训练模型（多输入需传入列表）
    history = model.fit(
        X_train_list, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val_list, y_val),
        callbacks=[early_stopping,reduce_lr],  # 启用回调
        verbose=1
    )

    # 4. 保存为官方推荐格式（解决警告）
    model.save("cr_cnn_relation_extraction.keras")
    print("模型已保存为 cr_cnn_relation_extraction.keras")
    return history, model

# ------------------- 启动训练 -------------------
history, trained_model = train_model(
    model,
    X_train_list=[X_word_train, X_head_pos_train, X_tail_pos_train],
    y_train=y_train,
    X_val_list=[X_word_val, X_head_pos_val, X_tail_pos_val],
    y_val=y_val,
    epochs=30,
    batch_size=32
)

# 步骤 6：模型评估与推理
def evaluate_model(model, X_test, y_test, label_map):
    """
    评估模型性能，输出Precision/Recall/F1
    :param model: 训练好的模型
    :param X_test/y_test: 测试集
    :param label_map: 标签映射字典
    """
    # 预测测试集
    y_pred = model.predict(X_test)
    y_pred_label = np.argmax(y_pred, axis=1)
    y_true_label = np.argmax(y_test, axis=1)

    # 计算指标（按类别）
    precision = precision_score(y_true_label, y_pred_label, average=None)
    recall = recall_score(y_true_label, y_pred_label, average=None)
    f1 = f1_score(y_true_label, y_pred_label, average=None)

    # 打印结果
    print("\n模型测试集评估结果：")
    for i, label in label_map.items():
        print(f"关系「{label}」：")
        print(f"  Precision: {precision[i]:.4f}, Recall: {recall[i]:.4f}, F1: {f1[i]:.4f}")
    # 宏观平均F1
    print(f"宏观平均F1: {f1_score(y_true_label, y_pred_label, average='macro'):.4f}")


# ------------------- 评估模型 -------------------
evaluate_model(
    trained_model,
    X_test=[X_word_test, X_head_pos_test, X_tail_pos_test],
    y_test=y_test,
    label_map=label_map
)

def infer_relations(model, test_pairs, word2idx, label_map, max_seq_len=128, max_dist=50):
    """适配多输入模型的推理函数"""
    # 1. 为测试样本生成特征
    X_word_infer = []
    X_head_pos_infer = []
    X_tail_pos_infer = []

    for pair in test_pairs:
        text_seq = pair["text_seq"]
        head = {"start": pair["head"]["local_start"], "end": pair["head"]["local_end"]}
        tail = {"start": pair["tail"]["local_start"], "end": pair["tail"]["local_end"]}

        # 词索引
        seq_idx = [word2idx.get(word, word2idx["[PAD]"]) for word in text_seq]
        seq_idx = pad_sequences([seq_idx], maxlen=max_seq_len, padding="post", truncating="post")[0]
        X_word_infer.append(seq_idx)

        # 位置索引
        head_center = (head["start"] + head["end"]) // 2
        tail_center = (tail["start"] + tail["end"]) // 2
        head_dist = []
        tail_dist = []

        for i in range(max_seq_len):
            d_h = min(max(i - head_center, -max_dist), max_dist) + max_dist
            d_t = min(max(i - tail_center, -max_dist), max_dist) + max_dist
            head_dist.append(d_h)
            tail_dist.append(d_t)

        X_head_pos_infer.append(head_dist)
        X_tail_pos_infer.append(tail_dist)

    # 2. 转为numpy数组
    X_word_infer = np.array(X_word_infer)
    X_head_pos_infer = np.array(X_head_pos_infer)
    X_tail_pos_infer = np.array(X_tail_pos_infer)

    # 3. 模型预测
    y_pred = model.predict([X_word_infer, X_head_pos_infer, X_tail_pos_infer], verbose=0)
    y_pred_label = np.argmax(y_pred, axis=1)
    y_true_label = np.argmax(y_test, axis=1)  # 需确保y_test与test_pairs对应

    # 4. 输出前10个三元组
    print("\n前10个测试样本的关系抽取结果：")
    for i in range(min(10, len(test_pairs))):
        t = test_pairs[i]
        pred_rel = label_map[y_pred_label[i]]
        true_rel = label_map[y_true_label[i]]
        prob = round(y_pred[i][y_pred_label[i]], 4)
        print(
            f"三元组{i + 1}：{t['head']['content']} → {pred_rel} → {t['tail']['content']}（真实：{true_rel}，概率：{prob}）")

    return test_pairs

# ------------------- 执行推理，输出三元组 -------------------
triples = infer_relations(
    trained_model,
    test_pairs=test_pairs,
    word2idx=word2idx,
    label_map=label_map,
    max_seq_len=128,
    max_dist=50
)