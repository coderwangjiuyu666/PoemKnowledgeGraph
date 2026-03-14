from neo4j import GraphDatabase
import pandas as pd

# Neo4j 数据库连接配置
uri = "neo4j://127.0.0.1:7687"  # 根据你的 Neo4j 配置修改
user = "neo4j"
password = "12345678"  # 根据你的 Neo4j 配置修改

# 建立数据库连接
driver = GraphDatabase.driver(uri, auth=(user, password))

# 文件列表
files = {
    '古诗文知识图谱数据10.12_朝代实体表.csv': '朝代实体',
    '古诗文知识图谱数据10.12_名句实体表.csv': '名句实体',
    '古诗文知识图谱数据10.12_诗文实体表.csv': '诗文实体',
    '古诗文知识图谱数据10.12_形式实体表.csv': '形式实体',
    '古诗文知识图谱数据10.12_主题实体表.csv': '主题实体',
    '古诗文知识图谱数据10.12_作者实体表.csv': '作者实体'
}

relationship_files = {
    '古诗文知识图谱数据10.12_创作关系表.csv': '创作',
    '古诗文知识图谱数据10.12_活跃于关系表.csv': '活跃于',
    '古诗文知识图谱数据10.12_具有主题关系表.csv': '具有主题',
    '古诗文知识图谱数据10.12_名句归属于关系表.csv': '名句归属于',
    '古诗文知识图谱数据10.12_属于形式关系表.csv': '属于形式'
}

# 实体类型对应的ID字段映射（关键修改：解决关系匹配问题）
id_field_mapping = {
    '朝代实体': '朝代ID',
    '名句实体': '名句ID',
    '诗文实体': '诗文ID',
    '形式实体': '形式ID',
    '主题实体': '主题ID',
    '作者实体': '作者ID'
}


def create_nodes(tx, label, df, id_field):
    """创建节点（防止重复）并返回创建数量"""
    count = 0
    total = len(df)
    for index, row in df.iterrows():
        properties = {k: v for k, v in row.items() if pd.notnull(v)}
        # 使用MERGE避免重复创建，基于唯一ID字段，并为节点指定变量名 n
        query = f"MERGE (n:{label} {{{id_field}: $id}}) ON CREATE SET {', '.join([f'n.{k} = ${k}' for k in properties.keys()])}"
        tx.run(query, id=properties[id_field], **properties)
        count += 1
        # 每100条打印一次进度
        if count % 100 == 0 or count == total:
            print(f"创建{label}节点: {count}/{total}")
    return count



def create_relationships(tx, relationship_type, df):
    """创建关系（防止重复）并返回创建数量"""
    count = 0
    total = len(df)
    for index, row in df.iterrows():
        start_id = row['起点ID']
        start_type = row['起点类型']
        end_id = row['终点ID']
        end_type = row['终点类型']

        # 获取对应实体的ID字段
        start_id_field = id_field_mapping[start_type]
        end_id_field = id_field_mapping[end_type]

        # 检查关系是否已存在，不存在则创建
        query = f"""
        MATCH (a:{start_type} {{{start_id_field}: $start_id}}), (b:{end_type} {{{end_id_field}: $end_id}})
        MERGE (a)-[r:{relationship_type}]->(b)
        ON CREATE SET r.created_at = timestamp()
        RETURN count(r) as created
        """
        result = tx.run(query, start_id=start_id, end_id=end_id)
        # 统计实际创建的关系数量
        if result.single()['created'] > 0:
            count += 1

        # 每100条打印一次进度
        if (index + 1) % 100 == 0 or (index + 1) == total:
            print(f"创建{relationship_type}关系: {count}/{total} (已跳过{index + 1 - count}条重复关系)")
    return count


# 创建实体节点
print("开始创建实体节点...")
with driver.session() as session:
    for file_path, label in files.items():
        print(f"\n处理文件: {file_path}")
        df = pd.read_csv(file_path)
        id_field = id_field_mapping[label]
        # 确保ID字段存在
        if id_field not in df.columns:
            print(f"错误：文件{file_path}中不存在ID字段{id_field}")
            continue
        # 执行节点创建
        created_count = session.execute_write(create_nodes, label, df, id_field)
        print(f"{label}节点创建完成，共创建{created_count}个节点")

# 创建关系
print("\n开始创建关系...")
with driver.session() as session:
    for file_path, relationship_type in relationship_files.items():
        print(f"\n处理文件: {file_path}")
        df = pd.read_csv(file_path)
        # 检查关系表必要字段
        required_fields = ['起点ID', '起点类型', '终点ID', '终点类型']
        if not all(field in df.columns for field in required_fields):
            print(f"错误：文件{file_path}缺少必要字段{required_fields}")
            continue
        # 执行关系创建
        created_count = session.execute_write(create_relationships, relationship_type, df)
        print(f"{relationship_type}关系创建完成，共创建{created_count}个关系")

# 关闭连接
driver.close()
print("\n所有导入操作完成，连接已关闭")