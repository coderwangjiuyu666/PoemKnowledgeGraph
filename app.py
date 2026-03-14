from flask import Flask, request, jsonify
from flask_cors import CORS
from neo4j import GraphDatabase, exceptions
import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import os
import re
from openai import OpenAI, APIError, Timeout, RateLimitError
import time
import logging
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Neo4j 配置
NEO4J_URI = "neo4j://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "12345678"

# 连接Neo4j
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
client = OpenAI(
    base_url="https://ark.cn-beijing.volces.com/api/v3",  # 文档指定API URL
    api_key=os.environ.get("ARK_API_KEY") or "02b95db9-3477-4c63-8918-ec1c0edfa2d8",  # 兼容环境变量与文档Key
)
def call_llm_api(prompt: str) -> str:
    """大模型API调用（适配文档提示词要求）"""
    try:
        completion = client.chat.completions.create(
            model="deepseek-v3-1-terminus",  # 官方指定模型
            messages=[
                {"role": "system",
                 "content": "你是一个将自然语言转换为Cypher查询的专家。"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.3,
            stream=False
        )
        return completion.choices[0].message.content.strip()
    except APIError as e:
        print(f"大模型API错误：{e}，提示词：{prompt[:20]}...")
        return "获取失败"
    except Timeout as e:
        print(f"大模型API超时：{e}，提示词：{prompt[:20]}...")
        return "获取失败"
    except RateLimitError as e:
        print(f"大模型API限流：{e}，提示词：{prompt[:20]}...")
        time.sleep(5)
        return call_llm_api(prompt) if prompt not in locals().get("retried_prompts", []) else "获取失败"
    except Exception as e:
        print(f"大模型调用未知错误：{e}，提示词：{prompt[:20]}...")
        return "获取失败"

# 实体ID字段映射（保留用于数据库交互）
id_field_mapping = {
    '朝代实体': '朝代ID',
    '名句实体': '名句ID',
    '诗文实体': '诗文ID',
    '形式实体': '形式ID',
    '主题实体': '主题ID',
    '作者实体': '作者ID'
}

# 实体名称字段映射
name_field_mapping = {
    '作者实体': '姓名',
    '朝代实体': '朝代名称',
    '诗文实体': '标题',
    '主题实体': '主题名称',
    '形式实体': '形式名称',
    '名句实体': '名句内容'
}


def generate_cypher(question):
    """调用大模型生成Cypher查询语句"""
    # 构建提示词
    prompt = f"""
    请将以下问题转换为Neo4j的Cypher查询语句，必须严格遵循数据库结构：

    数据库实体标签（必须完整使用，不可省略或修改）：
    - 作者实体（包含属性：姓名、作者ID、简介）
    - 朝代实体（包含属性：朝代名称、朝代ID、朝代跨度，朝代名称有先秦、两汉、魏晋、南北朝、隋代、唐代、五代、宋代、金朝、元代、明代、清代）
    - 诗文实体（包含属性：标题、诗文ID、原文）
    - 主题实体（包含属性：主题名称、主题ID、主题简介）
    - 形式实体（包含属性：形式名称、形式ID、形式简介）
    - 名句实体（包含属性：名句内容、名句ID）

    实体关系（必须严格使用）：
    - 作者实体 与 朝代实体 之间的关系：活跃于
    - 作者实体 与 诗文实体 之间的关系：创作
    - 名句实体 与 诗文实体 之间的关系：名句归属于
    - 诗文实体 与 形式实体 之间的关系：属于形式
    - 诗文实体 与 主题实体 之间的关系：具有主题

    生成规则：
    1. 必须使用上述完整实体标签（如用'作者实体'而非'作者'，'朝代实体'而非'朝代'）
    2. 必须使用正确的属性名称（如朝代实体的名称字段是'朝代名称'而非'name'）
    3. 只返回Cypher语句，不添加任何解释
    4. 若无法转换，返回空字符串

    问题：{question}
    """

    try:
        # 调用火山大模型
        cypher = call_llm_api(prompt)
        if cypher.startswith("```cypher"):
            cypher = cypher[:cypher.rfind("```")].replace("```cypher", "").strip()
        if cypher.endswith("```"):
            cypher = cypher[:-3].strip()
        print("生成的Cypher语句:", cypher)
        return cypher
    except Exception as e:
        print(f"调用大模型生成Cypher失败: {str(e)}")
        return ""


def execute_cypher(cypher):
    """执行Cypher语句并返回结果"""
    try:
        with driver.session() as session:
            result = session.run(cypher, timeout=5000)
            records = [dict(record) for record in result]
            print(f"Cypher执行成功，返回{len(records)}条记录")
            return records
    except exceptions.Neo4jError as e:
        print(f"Cypher执行错误: {str(e)}")
        return None
    except Exception as e:
        print(f"数据库操作异常: {str(e)}")
        return None


def format_answer(records):
    """格式化查询结果为自然语言回答"""
    if not records:
        return "未找到相关信息"

    answer_parts = []
    for i, record in enumerate(records, 1):
        names = []
        for value in record.values():  # 直接遍历记录中的实体值
            # 尝试获取实体类型
            entity_type = None
            if hasattr(value, 'labels'):
                labels = list(value.labels)
                if labels:
                    entity_type = labels[0]

            # 获取实体名称
            if entity_type and entity_type in name_field_mapping:
                name_field = name_field_mapping[entity_type]
                if hasattr(value, 'get') and name_field in value:
                    name = value.get(name_field)  # 直接提取名称（如诗人姓名）
                else:
                    name = str(value)
            else:
                name = str(value)

            names.append(name)  # 收集当前记录中的所有实体名称

        # 只显示序号+名称，去掉多余的键名
        answer_parts.append(f"{i}. " + ", ".join(names))

    return "\n".join(answer_parts)

@app.route('/api/answer', methods=['GET'])
def answer_question():
    question = request.args.get('question', '').strip()
    if not question:
        return jsonify({'answer': '请输入问题'})

    print(f"收到问题: {question}")

    # 1. 调用大模型生成Cypher
    cypher = generate_cypher(question)
    if not cypher:
        return jsonify({'answer': '无法生成有效的查询语句，请尝试其他问题'})

    # 2. 执行Cypher查询
    records = execute_cypher(cypher)
    if records is None:
        return jsonify({'answer': '查询执行失败，请检查问题或稍后再试'})

    # 3. 格式化结果
    answer = format_answer(records)

    return jsonify({
        'answer': answer,
        'cypher': cypher  # 返回生成的Cypher，方便调试
    })


@app.route('/api/knowledge-graph', methods=['GET'])
def get_knowledge_graph():
    """保留知识图谱可视化接口"""
    entity_types = request.args.get('entityTypes', '').split(',')
    relationship_types = request.args.get('relationshipTypes', '').split(',')
    limit = int(request.args.get('limit', 100))

    if entity_types == ['']:
        entity_types = []
    if relationship_types == ['']:
        relationship_types = []

    # 添加默认实体类型，确保即使没有选择也能返回数据
    if not entity_types:
        entity_types = ['作者实体']  # 默认返回作者实体

    final_entity_types = set(entity_types)
    if relationship_types:
        relationship_entity_mapping = {
            '创作': ['作者实体', '诗文实体'],
            '活跃于': ['作者实体', '朝代实体'],
            '具有主题': ['诗文实体', '主题实体'],
            '名句归属于': ['名句实体', '诗文实体'],
            '属于形式': ['诗文实体', '形式实体']
        }
        for rel in relationship_types:
            if rel in relationship_entity_mapping:
                final_entity_types.update(relationship_entity_mapping[rel])

    final_entity_types = list(final_entity_types)
    print(f"查询实体类型: {final_entity_types}, 关系类型: {relationship_types}, 限制: {limit}")

    try:
        with driver.session() as session:
            if not relationship_types:
                # 只查询实体，不查询关系 - 使用原始方式构建查询
                query = """
                MATCH (n)
                WHERE n:{}
                RETURN n
                LIMIT $limit
                """.format('|'.join(final_entity_types))
                print(f"执行查询: {query}")
                result = session.run(query, limit=limit,timeout=5000)

                # 处理结果 - 使用原始方式
                nodes = {}
                for record in result:
                    node = record['n']
                    node_label = list(node.labels)[0]
                    node_id = node[id_field_mapping[node_label]]
                    if node_id not in nodes:
                        nodes[node_id] = format_node(node)

                nodes_list = list(nodes.values())
                print(f"返回节点数量: {len(nodes_list)}")
                return jsonify({
                    'nodes': nodes_list,
                    'links': []
                })
            else:
                # 查询实体和关系 - 使用原始方式构建查询
                query = """
                MATCH (n)-[r:{}]-(m)
                WHERE n:{} AND m:{}
                RETURN n, r, m
                LIMIT $limit
                """.format(
                    '|'.join(relationship_types),
                    '|'.join(final_entity_types),
                    '|'.join(final_entity_types)
                )
                print(f"执行查询: {query}")
                result = session.run(query, limit=limit,timeout=5000)

                # 处理结果 - 使用原始方式
                nodes = {}
                links = []
                for record in result:
                    # 处理源节点
                    node = record['n']
                    node_label = list(node.labels)[0]
                    node_id = node[id_field_mapping[node_label]]
                    if node_id not in nodes:
                        nodes[node_id] = format_node(node)

                    # 处理目标节点
                    related_node = record['m']
                    related_label = list(related_node.labels)[0]
                    related_id = related_node[id_field_mapping[related_label]]
                    if related_id not in nodes:
                        nodes[related_id] = format_node(related_node)

                    # 处理关系
                    rel = record['r']
                    links.append({
                        'source': node_id,
                        'target': related_id,
                        'name': rel.type
                    })

                # 转换为列表
                nodes_list = list(nodes.values())

                # 限制节点数量
                if len(nodes_list) > limit:
                    nodes_list = nodes_list[:limit]
                    # 过滤不在节点列表中的关系
                    node_ids = [n['id'] for n in nodes_list]
                    links = [l for l in links if l['source'] in node_ids and l['target'] in node_ids]

                print(f"返回节点数量: {len(nodes_list)}, 关系数量: {len(links)}")
                return jsonify({
                    'nodes': nodes_list,
                    'links': links
                })
    except Exception as e:
        print(f"知识图谱查询错误: {str(e)}")
        return jsonify({'error': str(e)}), 500


# 同时，让我们简化format_node函数，确保它不会过滤掉有效数据
def format_node(node):
    """格式化节点数据"""
    try:
        # 获取标签
        labels = list(node.labels)
        label = labels[0] if labels else ''
        
        # 获取ID字段
        id_field = id_field_mapping.get(label, '')
        
        # 获取属性
        properties = dict(node.items())
        
        # 确定节点名称
        name_fields = ['姓名', '朝代名称', '标题', '名句内容', '形式名称', '主题名称']
        name = properties.get(next((f for f in name_fields if f in properties), ''), str(node.id))
        
        # 返回节点对象
        return {
            'id': properties[id_field] if id_field and id_field in properties else str(node.id),
            'name': name,
            'label': label,
            'idField': id_field,
            'itemStyle': {
                'color': get_entity_color(label)
            },
            'properties': properties,
            'dataIndex': 0  # 确保包含dataIndex字段，避免前端报错
        }
    except Exception as e:
        print(f"节点格式化错误: {str(e)}")
        # 即使出错也返回基本结构，避免前端无法处理
        return {
            'id': 'error_node_' + str(time.time()),
            'name': 'Error Node',
            'label': 'Error',
            'idField': 'id',
            'itemStyle': {'color': '#FF0000'},
            'properties': {},
            'dataIndex': 0
        }
def get_entity_color(entity_type):
    """获取实体类型对应的颜色"""
    colors = {
        '作者实体': '#165DFF',
        '诗文实体': '#722ED1',
        '朝代实体': '#F53F3F',
        '名句实体': '#FF7D00',
        '形式实体': '#0FC6C2',
        '主题实体': '#7BC616'
    }
    return colors.get(entity_type, '#888888')


@app.route('/api/search', methods=['GET'])
def search_entities():
    """搜索包含指定文本的实体"""
    term = request.args.get('term', '').strip()
    if not term:
        return jsonify({'entities': []})

    print(f"搜索实体: {term}")

    try:
        with driver.session() as session:
            # 构建搜索所有实体类型的Cypher查询
            query = """
            MATCH (n)
            WHERE 
                (n:作者实体 AND (n.姓名 CONTAINS $term OR n.简介 CONTAINS $term)) OR
                (n:诗文实体 AND (n.标题 CONTAINS $term OR n.原文 CONTAINS $term)) OR
                (n:朝代实体 AND (n.朝代名称 CONTAINS $term OR n.朝代跨度 CONTAINS $term)) OR
                (n:名句实体 AND n.名句内容 CONTAINS $term) OR
                (n:形式实体 AND (n.形式名称 CONTAINS $term OR n.形式简介 CONTAINS $term)) OR
                (n:主题实体 AND (n.主题名称 CONTAINS $term OR n.主题简介 CONTAINS $term))
            RETURN n
            LIMIT 50
            """

            result = session.run(query, term=term, timeout=5000)
            entities = [format_node(record['n']) for record in result]

            print(f"搜索到 {len(entities)} 个匹配实体")
            return jsonify({'entities': entities})
    except Exception as e:
        print(f"搜索实体错误: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/entity-relations', methods=['GET'])
def get_entity_relations():
    """获取指定实体及其直接关联的节点"""
    entity_id = request.args.get('id', '').strip()
    entity_label = request.args.get('label', '').strip()

    if not entity_id or not entity_label:
        return jsonify({'error': '缺少实体ID或标签'}), 400

    print(f"获取实体关系: {entity_label} - {entity_id}")

    try:
        with driver.session() as session:
            # 获取实体及其直接关系的节点
            query = f"""
            MATCH (n:{entity_label})-[r]-(m)
            WHERE n.`{id_field_mapping[entity_label]}` = $entity_id
            RETURN n, r, m
            """

            result = session.run(query, entity_id=entity_id, timeout=5000)

            nodes = {}
            links = []

            # 添加中心节点
            center_node = None
            for record in result:
                if not center_node:
                    center_node = format_node(record['n'])
                    nodes[center_node['id']] = center_node

                # 处理关联节点
                related_node = format_node(record['m'])
                if related_node['id'] not in nodes:
                    nodes[related_node['id']] = related_node

                # 处理关系
                rel = record['r']
                links.append({
                    'source': center_node['id'],
                    'target': related_node['id'],
                    'name': rel.type
                })

            # 如果没有找到关系，至少返回中心节点
            if not center_node:
                # 单独查询中心节点
                query = f"""
                MATCH (n:{entity_label})
                WHERE n.`{id_field_mapping[entity_label]}` = $entity_id
                RETURN n
                """
                result = session.run(query, entity_id=entity_id)
                for record in result:
                    center_node = format_node(record['n'])
                    nodes[center_node['id']] = center_node

            nodes_list = list(nodes.values())

            print(f"返回关联节点数量: {len(nodes_list)}, 关系数量: {len(links)}")
            return jsonify({
                'nodes': nodes_list,
                'links': links
            })
    except Exception as e:
        print(f"获取实体关系错误: {str(e)}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=False,threaded=True)