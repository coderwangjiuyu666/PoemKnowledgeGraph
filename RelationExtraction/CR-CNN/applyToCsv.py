import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from collections import defaultdict


def extract_entities_from_text(text):
    """
    从文本中提取实体（这里使用简单规则示例，实际应用中可能需要NER模型）
    假设文本格式：[诗名]作者：[作者名]朝代：[朝代名]
    """
    entities = []

    # 简单规则提取（实际应用中建议使用成熟的NER模型）
    if "作者：" in text and "朝代：" in text:
        # 提取诗名
        poem_end = text.find("作者：")
        if poem_end > 0:
            poem_text = text[:poem_end].strip()
            entities.append({
                "type": "Poem",
                "content": poem_text,
                "local_start": 0,
                "local_end": len(poem_text) - 1  # 简化处理，实际应按token计算
            })

        # 提取作者
        author_start = text.find("作者：") + 3
        author_end = text.find("朝代：")
        if author_start < author_end:
            author_text = text[author_start:author_end].strip()
            entities.append({
                "type": "Author",
                "content": author_text,
                "local_start": len(poem_text) + 3,  # 简化处理
                "local_end": len(poem_text) + 3 + len(author_text) - 1
            })

        # 提取朝代
        dynasty_start = text.find("朝代：") + 3
        dynasty_text = text[dynasty_start:].strip()
        entities.append({
            "type": "Dynasty",
            "content": dynasty_text,
            "local_start": len(poem_text) + 3 + len(author_text) + 3,  # 简化处理
            "local_end": len(text) - 1
        })

    return entities


def process_real_text(text, model, word2idx, label_map, max_seq_len=128, max_dist=50):
    """处理单条真实文本，提取实体并预测关系"""
    # 分词（实际应用中可能需要更复杂的分词器）
    tokens = list(text)  # 简单按字符分词，实际可使用jieba等工具

    # 提取实体
    entities = extract_entities_from_text(text)
    if len(entities) < 3:  # 确保有诗名、作者、朝代三种实体
        return []

    entity_dict = defaultdict(list)
    for ent in entities:
        entity_dict[ent["type"]].append(ent)

    # 生成可能的实体对
    pairs = []
    if entity_dict.get("Author") and entity_dict.get("Poem") and entity_dict.get("Dynasty"):
        author = entity_dict["Author"][0]
        poem = entity_dict["Poem"][0]
        dynasty = entity_dict["Dynasty"][0]

        # 生成所有可能的实体对
        pairs.append({"text_seq": tokens, "head": author, "tail": dynasty})
        pairs.append({"text_seq": tokens, "head": author, "tail": poem})
        pairs.append({"text_seq": tokens, "head": poem, "tail": dynasty})
        pairs.append({"text_seq": tokens, "head": dynasty, "tail": poem})

    # 特征处理
    X_word = []
    X_head_pos = []
    X_tail_pos = []

    for pair in pairs:
        text_seq = pair["text_seq"]
        head = {"start": pair["head"]["local_start"], "end": pair["head"]["local_end"]}
        tail = {"start": pair["tail"]["local_start"], "end": pair["tail"]["local_end"]}

        # 词索引
        seq_idx = [word2idx.get(word, word2idx["[PAD]"]) for word in text_seq]
        seq_idx = pad_sequences([seq_idx], maxlen=max_seq_len, padding="post", truncating="post")[0]
        X_word.append(seq_idx)

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

        X_head_pos.append(head_dist)
        X_tail_pos.append(tail_dist)

    if not X_word:
        return []

    # 模型预测
    X_word = np.array(X_word)
    X_head_pos = np.array(X_head_pos)
    X_tail_pos = np.array(X_tail_pos)

    y_pred = model.predict([X_word, X_head_pos, X_tail_pos], verbose=0)
    y_pred_label = np.argmax(y_pred, axis=1)

    # 整理结果
    results = []
    for i, pair in enumerate(pairs):
        head_ent = pair["head"]["content"]
        tail_ent = pair["tail"]["content"]
        rel_type = label_map[y_pred_label[i]]
        confidence = float(y_pred[i][y_pred_label[i]])

        # 只保留有意义的关系（过滤"无关系"或低置信度结果）
        if rel_type != "无关系" and confidence > 0.5:
            results.append({
                "head_entity": head_ent,
                "relation": rel_type,
                "tail_entity": tail_ent,
                "confidence": confidence,
                "original_text": text
            })

    return results


def batch_process_texts(texts, model, word2idx, label_map, output_file="relation_results.csv"):
    """批量处理文本并导出结果到CSV"""
    all_results = []

    for i, text in enumerate(texts):
        print(f"处理文本 {i + 1}/{len(texts)}")
        results = process_real_text(text, model, word2idx, label_map)
        all_results.extend(results)

    # 导出到CSV
    df = pd.DataFrame(all_results)
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"结果已导出到 {output_file}，共 {len(all_results)} 条关系")

    return df

def load_texts_from_txt(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]
    return texts
# ------------------- 实际使用示例 -------------------
if __name__ == "__main__":
    # 加载训练好的模型
    model = tf.keras.models.load_model("cr_cnn_relation_extraction.keras")

    # 假设我们已经有了word2idx和label_map（实际应用中需要从训练过程中保存和加载）
    # 这里只是示例，实际应使用训练时的词汇表
    label_map = {0: "无关系", 1: "属于", 2: "创作"}

    # 示例文本数据
    sample_texts =load_texts_from_txt("input_texts.txt")

    # 批量处理并导出结果
    result_df = batch_process_texts(
        texts=sample_texts,
        model=model,
        word2idx=word2idx,  # 这里使用训练时的word2idx
        label_map=label_map,
        output_file="poetry_relation_results.csv"
    )

    # 打印前几行结果
    print("\n抽取的关系结果：")
    print(result_df.head())