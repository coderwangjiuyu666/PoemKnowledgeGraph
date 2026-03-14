# 古诗文知识图谱可视化系统

## 项目简介

本项目以**Neo4j**为图数据库核心，结合**Python**爬虫、**BiLSTM-CRF**实体抽取、**CR-CNN**关系抽取（综合比较考虑了CR-CNN、Attention CNNs、Att-BLSTM）、**Flask**后端服务、**ECharts**前端可视化及大模型API自然语言转Cypher能力，构建了覆盖**6类核心实体**、**5类核心关系**的古诗文知识图谱，并提供图谱可视化、智能问答、实体搜索三大核心功能。

## 知识图谱设计

### 实体类型（6类）

| 实体类型 | 核心属性                   | 说明                            |
| :------- | :------------------------- | :------------------------------ |
| 作者实体 | 作者ID、姓名、简介         | 古诗文作者的唯一标识与基础信息  |
| 朝代实体 | 朝代ID、朝代名称、时间跨度 | 涵盖先秦至清代12个核心朝代      |
| 诗文实体 | 诗文ID、标题、原文         | 古诗文作品的核心信息            |
| 名句实体 | 名句ID、名句内容           | 古诗文经典名句提取              |
| 主题实体 | 主题ID、主题名称、主题简介 | 送别、爱国、思乡等50+古诗文主题 |
| 形式实体 | 形式ID、形式名称、形式简介 | 诗、词、曲三大核心文体          |

### 关系类型（5类）

| 关系类型   | 关联实体            | 说明                      |
| :--------- | :------------------ | :------------------------ |
| 创作       | 作者实体 → 诗文实体 | 作者创作某篇古诗文        |
| 活跃于     | 作者实体 → 朝代实体 | 作者活跃的历史朝代        |
| 具有主题   | 诗文实体 → 主题实体 | 古诗文所属的情感/内容主题 |
| 名句归属于 | 名句实体 → 诗文实体 | 名句对应的原诗文作品      |
| 属于形式   | 诗文实体 → 形式实体 | 古诗文所属的文体形式      |

## 功能展示

### 1. 知识图谱可视化

- 只选取节点：以选取“作者”节点为例，系统默认限制节点数量为100，也可以自行调整，点击“查询知识图谱”后，结果如下：

  ![image-20260314195756222](C:\Users\wangjy\AppData\Roaming\Typora\typora-user-images\image-20260314195756222.png)

  将光标放在节点上，能看到节点信息

  <img src="C:\Users\wangjy\AppData\Roaming\Typora\typora-user-images\image-20260314195910523.png" alt="image-20260314195910523" style="zoom:50%;" />

点击节点之后，会在知识图谱上方展示节点的属性信息

<img src="C:\Users\wangjy\AppData\Roaming\Typora\typora-user-images\image-20260314200001964.png" alt="image-20260314200001964" style="zoom: 50%;" />

系统支持多个仅节点的查询：

![image-20260314200130671](C:\Users\wangjy\AppData\Roaming\Typora\typora-user-images\image-20260314200130671.png)

- 选取关系：选取某种关系后，与该关系有关的节点自动被选取，点击“查询知识图谱”之后，节点及其关系便可展示到前端

  ![image-20260314200324438](C:\Users\wangjy\AppData\Roaming\Typora\typora-user-images\image-20260314200324438.png)

把鼠标光标放在某一实体上，系统将聚焦于该实体，展示与其有一跳关系的节点

![image-20260314200418328](C:\Users\wangjy\AppData\Roaming\Typora\typora-user-images\image-20260314200418328.png)

在页面左下角，系统还展示了当前画面的节点数量和关系数量，本系统目前最多包含1400多个节点和6000多条关系，但为了美观，仅支持拉取节点数量到1000

![image-20260314200521811](C:\Users\wangjy\AppData\Roaming\Typora\typora-user-images\image-20260314200521811.png)

### 2. 自然语言智能问答

- 支持用户以自然语言提问古诗文相关问题（如：*苏轼写过哪些关于送别的作品*），调用LLM API解析为Cypher语句之后，在数据库里面查询，并把结果渲染到前端

  <img src="C:\Users\wangjy\AppData\Roaming\Typora\typora-user-images\image-20260314200744352.png" alt="image-20260314200744352" style="zoom:67%;" />

  若用户输入的语句不符合本系统要求，系统也会返回以下用户友好信息

  <img src="C:\Users\wangjy\AppData\Roaming\Typora\typora-user-images\image-20260314200922354.png" alt="image-20260314200922354" style="zoom: 67%;" />

### 3. 实体关键词搜索

- 支持模糊搜索：输入关键词可匹配所有实体的核心属性（如作者姓名/简介、诗文标题/原文、名句内容等）

  ![image-20260314201041948](C:\Users\wangjy\AppData\Roaming\Typora\typora-user-images\image-20260314201041948.png)

- 搜索结果以候选栏展示，点击选中实体后，自动展示该实体的详细属性及所有直接关联的节点与关系

<img src="C:\Users\wangjy\AppData\Roaming\Typora\typora-user-images\image-20260314201112559.png" alt="image-20260314201112559" style="zoom:67%;" />

## 技术栈

### 后端/算法

- 编程语言：Python 3.8+
- 数据采集：Requests、BeautifulSoup4（爬虫）
- 实体抽取：PyTorch、BiLSTM-CRF
- 关系抽取：PyTorch、CR-CNN/Attention CNNs/Att-BLSTM
- 后端框架：Flask、Flask-CORS
- 图数据库：Neo4j + Neo4j Python Driver
- 大模型对接：OpenAI API（适配火山方舟deepseek-v3-1-terminus/doubao-1-5-lite-32k）
- 数据处理：Pandas、NumPy、Scikit-learn

### 前端

- 基础框架：HTML、CSS、JavaScript
- 样式：Tailwind CSS
- 可视化：ECharts（知识图谱力导向图）
- 交互：AJAX

### 开发工具

- 数据标注：doccano
- 依赖管理：pip

## 系统架构

系统采用**四层分层架构**，实现数据、算法、服务、展示的解耦与协同：

1. **数据存储层**：Neo4j图数据库，存储6类实体和5类关系的结构化知识图谱数据
2. **算法层**：实现实体抽取（BiLSTM-CRF）、关系抽取（CR-CNN）、自然语言转Cypher（大模型API）
3. **后端服务层**：Flask构建RESTful API，提供图谱查询、智能问答、实体搜索接口，解决跨域问题
4. **前端展示层**：基于ECharts实现知识图谱可视化，支持多维度筛选、节点交互、结果展示

**核心流程**：

- 数据采集：爬虫爬取古诗文网+大模型API补充实体属性 → 数据清洗与标准化
- 知识构建：实体抽取→关系抽取→CSV数据导出→Neo4j图谱导入
- 应用服务：前端请求→后端API处理（Cypher生成/执行）→ 结果格式化→前端可视化/展示

## 项目结构

```Plain
gushiwen-knowledge-graph/
├── app.py                # 后端主服务：Flask API、Neo4j交互、大模型调用
├── Crawl.py              # 数据采集：爬虫爬取古诗文网+大模型API补充实体属性
├── importToNeo4j.py      # 图谱导入：将CSV实体/关系数据导入Neo4j
├── 古诗文知识图谱数据10.12.xlsx # 实体/关系原始数据（Excel）
├── 各实体/关系CSV文件/   # 导入Neo4j的标准化CSV数据
├── entity_extraction/    # 实体抽取模块：BiLSTM-CRF模型训练与推理
├── relation_extraction/  # 关系抽取模块：CR-CNN等模型训练与推理
├── front/                # 前端代码：可视化、交互、页面展示
└── README.md             # 项目说明文档
```

## 环境配置与运行

### 1. 环境依赖安装

```Bash
pip install -r requirements.txt
```

### 2. 图数据库配置（Neo4j）

1. 安装并启动Neo4j（推荐4.4+版本），修改`app.py`和`importToNeo4j.py`中的Neo4j配置：

   ```python
   NEO4J_URI = "neo4j://localhost:7687"
   NEO4J_USER = "你的Neo4j用户名"
   NEO4J_PASSWORD = "你的Neo4j密码"
   ```

2. 确保Neo4j服务正常运行，无端口占用。

### 3. 大模型API配置

修改`app.py`和`Crawl.py`中的大模型API配置（适配火山方舟）：

```Python
client = OpenAI(
    base_url="https://ark.cn-beijing.volces.com/api/v3",  # 火山方舟API地址
    api_key="你的火山方舟API_KEY"  # 替换为自己的API_KEY，也可通过环境变量ARK_API_KEY设置
)
```

### 4. 图谱导入

1. 开启Neo4j的一个instance：

2. 执行导入脚本将CSV数据导入Neo4j，构建知识图谱：

   ```bash
   python importToNeo4j.py
   ```

### 5. 启动后端服务

```Bash
python app.py
```

服务将运行在`http://0.0.0.0:5000`

### 6. 启动前端

直接用浏览器打开`index.html`，前端将自动对接后端API。

## 核心接口说明

### 1. 知识图谱查询接口

- **URL**：`/api/knowledge-graph`
- **Method**：GET
- **Params**：
  - `entityTypes`：实体类型，多个以逗号分隔（如`作者实体,诗文实体`）
  - `relationshipTypes`：关系类型，多个以逗号分隔（如`创作,活跃于`）
  - `limit`：节点数量限制，默认100
- **Return**：JSON格式，包含`nodes`（节点列表）、`links`（关系列表）

### 2. 智能问答接口

- **URL**：`/api/answer`
- **Method**：GET
- **Params**：`question`（用户自然语言问题）
- **Return**：JSON格式，包含`answer`（自然语言答案）、`cypher`（生成的Cypher语句）

### 3. 实体搜索接口

- **URL**：`/api/search`
- **Method**：GET
- **Params**：`term`（搜索关键词）
- **Return**：JSON格式，包含`entities`（匹配的实体列表）

### 4. 实体关联关系查询接口

- **URL**：`/api/entity-relations`
- **Method**：GET
- **Params**：
  - `id`：实体ID
  - `Label`：实体类型
- **Return**：JSON格式，包含`nodes`（实体及关联节点）、`links`（关联关系）

## 关键算法实现

### 1. 实体抽取（BiLSTM-CRF）

- 采用**BIO格式**对古诗文文本进行实体标注（B-实体、I-实体、O-非实体）
- 模型由**嵌入层+双向LSTM层+全连接层+CRF层**组成，捕捉序列上下文特征与标签依赖关系
- 加入**位置特征**（句首、标点）辅助实体边界识别，测试集F1分数达0.9433

### 2. 关系抽取（CR-CRF）

- 设计**CR-CNN、Attention CNNs、Att-BLSTM**三种模型对比实验，最终选择CR-CNN（避免过拟合，泛化能力更强）
- 融合**词嵌入+实体位置嵌入**，采用多尺寸卷积核（3/5/7）提取局部语义特征，全局最大池化后分类
- 支持三类关系分类：无关系、创作、活跃于，测试集宏观F1分数达0.9587

### 3. 自然语言转Cypher

- 基于大模型**deepseek-v3-1-terminus**，构建精准的提示词模板，约束大模型生成符合Neo4j语法的Cypher语句
- 对生成的Cypher语句进行后处理（去除代码块标记、空格清洗），保证执行有效性
- 加入异常重试机制，对限流、超时等API异常进行处理