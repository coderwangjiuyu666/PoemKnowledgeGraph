import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import hashlib

import urllib.parse
# 导入官方示例中的OpenAI客户端及异常类（用于API调用）
from openai import OpenAI, APIError, Timeout, RateLimitError

# ============================ 1. 全局配置（基于最新文档要求）===========================
# 1.1 爬虫基础配置（文档指定：仅爬前两页）
BASE_URL = "https://www.gushiwen.cn"
PAGE_RANGE = range(1, 3)  # 仅爬前两页（文档强制要求）
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Referer": "https://www.gushiwen.cn/"
}

# 1.2 实体列表（严格遵循文档指定）
DYNastIES = ["先秦", "两汉", "魏晋", "南北朝", "隋代", "唐代", "五代", "宋代", "金朝", "元代", "明代", "清代"]
TOPICS = ["送别", "劝学", "边塞", "儿童", "春天", "夏天", "秋天", "冬天", "悲愤", "悼亡", "咏怀", "爱国", "思乡",
          "咏物", "爱情", "田园", "民歌", "民谣", "山水", "怀古", "咏史", "散文", "闺怨", "抒情", "赞美", "咏柳",
          "读书", "秋思", "哲理", "离别", "梅花", "叙事", "写雪", "写景", "月亮", "长诗", "励志", "战争", "荷花",
          "题画", "感恩", "动物", "散曲", "感怀", "饮酒", "落花", "桃花", "写雨", "青春", "写山", "论诗", "游仙",
          "节日", "春节", "元宵节", "寒食节", "清明节", "端午节", "七夕节", "中秋节", "重阳节", "托物言志"]
FORMS = ["诗", "词", "曲", "其他"]

# 1.3 初始化OpenAI客户端（文档指定API信息）
client = OpenAI(
    base_url="https://ark.cn-beijing.volces.com/api/v3",  # 文档指定API URL
    api_key=os.environ.get("ARK_API_KEY") or "02b95db9-3477-4c63-8918-ec1c0edfa2d8",  # 兼容环境变量与文档Key
)

# 1.4 知识图谱表格初始化（作者实体表移除“字/号”，文档要求）
entity_tables = {
    "朝代实体表": pd.DataFrame(columns=["朝代ID", "朝代名称", "时间跨度"]),
    "名句实体表": pd.DataFrame(columns=["名句ID", "名句内容"]),
    "主题实体表": pd.DataFrame(columns=["主题ID", "主题名称", "主题简介"]),
    "形式实体表": pd.DataFrame(columns=["形式ID", "形式名称"])
}

# ============================ 2. 工具函数（适配文档修改）===========================
def get_entity_id(content: str) -> str:
    """生成唯一实体ID（哈希去重）"""
    return hashlib.md5(content.encode("utf-8")).hexdigest()[:16]


def fetch_page(url: str) -> BeautifulSoup:
    """爬虫页面请求（反爬延时+异常捕获）"""
    try:
        time.sleep(1)  # 避免高频请求被封
        response = requests.get(url, headers=HEADERS, timeout=20)
        response.raise_for_status()
        return BeautifulSoup(response.text, "html.parser")
    except Exception as e:
        print(f"页面请求失败：{url}，错误：{str(e)}")
        return None


def call_llm_api(prompt: str) -> str:
    """大模型API调用（适配文档提示词要求）"""
    try:
        completion = client.chat.completions.create(
            model="doubao-1-5-lite-32k-250115",  # 官方指定模型
            messages=[
                {"role": "system",
                 "content": "你是古诗文知识图谱属性提取助手，严格按提示词格式返回，不添加额外内容，基于史实回答。"},
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


def add_entity(table_name: str, entity_data: dict):
    """添加实体到表格（去重）"""
    table = entity_tables[table_name]
    if "ID" in entity_data and entity_data["ID"] in table["ID"].values:
        return
    table = pd.concat([table, pd.DataFrame([entity_data])], ignore_index=True)
    entity_tables[table_name] = table

def crawl_dynasties():
    llm_prompt = "请分别给出先秦、两汉、魏晋、南北朝、隋代、唐代、五代、宋代、金朝、元代、明代、清代的开始年份和结束年份，如“a年-b年”，以英文分号间隔"
    llm_result = call_llm_api(llm_prompt)

    processed_result = llm_result.replace("；", ";").replace(" ", "").replace("　", "")
    dynasty_spans = processed_result.split(";") if ";" in processed_result else []
    while len(dynasty_spans) < 12:
        dynasty_spans.append("获取失败")
    for i, dynasty in enumerate(DYNastIES):
        span_item = dynasty_spans[i].strip()
        if span_item == "获取失败":
            span = "获取失败"
        else:
            if "：" in span_item:
                span = span_item.split("：")[-1].strip()
            elif ":" in span_item:
                span = span_item.split(":")[-1].strip()
            else:
                span = span_item if "-" in span_item else "获取失败"
        add_entity("朝代实体表", {
            "朝代ID": get_entity_id(dynasty),
            "朝代名称": dynasty,
            "时间跨度": span
        })
        print(f"已添加朝代实体：{dynasty}（时间跨度：{span}）")

def crawl_quotes():
    for page in PAGE_RANGE:
        if page == 1:
            quote_url = f"{BASE_URL}/mingjus/default.aspx?page=1&tstr=&astr=&cstr=&xstr=诗文"
        else:
            quote_url = f"{BASE_URL}/mingjus/default.aspx?page={page}&tstr=&astr=&cstr=&xstr=诗文"
        soup = fetch_page(quote_url)
        if not soup:
            print(f"名句页面第{page}页请求失败")
            continue
        quote_cards = soup.find_all("div", class_="cont", style=lambda x: x and "margin-top: 12px" in x)
        for card in quote_cards:
            # 1. 提取名句内容
            quote_tag = card.find("a", href=lambda x: x and "/mingju/juv_" in x)
            if not quote_tag:
                continue
            quote_content = quote_tag.text.strip()
            quote_id = get_entity_id(quote_content)

            # 2. 添加名句实体
            add_entity("名句实体表", {
                "名句ID": quote_id,
                "名句内容": quote_content
            })
            print(f"已添加名句实体：{quote_content[:20]}...")


def crawl_topics():
    PAGE_LIMIT = [1, 2]  # 硬编码前两页
    for topic_name in TOPICS:
        topic_id = get_entity_id(topic_name)
        encoded_topic = urllib.parse.quote(topic_name)

        for page in PAGE_LIMIT:
            # 构建URL（确保只爬1、2页）
            if page == 1:
                topic_url = f"{BASE_URL}/shiwens/default.aspx?tstr={encoded_topic}&astr=&cstr=&xstr="
            else:
                topic_url = f"{BASE_URL}/shiwens/default.aspx?page={page}&tstr={encoded_topic}&astr=&cstr=&xstr="

            soup = fetch_page(topic_url)
            if not soup:
                print(f"主题[{topic_name}]第{page}页请求失败，跳过")
                continue

            if topic_name not in entity_tables["主题实体表"]["主题名称"].values:
                llm_prompt = f"请分别给出50字左右的{topic_name}古诗文内容主题的简介，以英文分号分隔"
                topic_intro = call_llm_api(llm_prompt).replace(";", "").strip()
                add_entity("主题实体表", {
                    "主题ID": topic_id,
                    "主题名称": topic_name,
                    "主题简介": topic_intro
                })
                print(f"已添加主题实体：{topic_name}")
            else:
                print(f"主题实体{topic_name}已存在，跳过")

def crawl_forms():
    """修改后：仅新增诗/词/曲实体并添加100字简介"""
    # 仅保留诗/词/曲三种形式
    FORMS = ["诗", "词", "曲"]
    for form_name in FORMS:
        form_id = get_entity_id(form_name)
        # 检查实体是否已存在，不存在则创建并获取简介
        if form_name not in entity_tables["形式实体表"]["形式名称"].values:
            # 调用大模型获取100字简介
            llm_prompt = f"请给出{form_name}的100字左右简介，说明其特点、起源和主要特征"
            form_intro = call_llm_api(llm_prompt).strip()
            add_entity("形式实体表", {
                "形式ID": form_id,
                "形式名称": form_name,
                "形式简介": form_intro  # 新增简介字段
            })
            print(f"已添加形式实体：{form_name}（简介：{form_intro[:30]}...）")
        else:
            print(f"形式实体{form_name}已存在，跳过")

# ============================ 4. 主函数（文档指定操作顺序：④→③→⑤→⑥→⑦）===========================
def main():
    print("=" * 50)
    print("开始爬取朝代实体...")
    crawl_dynasties()
    print("=" * 50)
    print("开始爬取形式实体...")
    crawl_forms()
    print("\n" + "=" * 50)
    print("开始执行文档操作⑤：爬取名句实体")
    crawl_quotes()
    print("\n" + "=" * 50)
    print("开始执行文档操作⑥：爬取主题实体")
    crawl_topics()

    # 保存结果（文档要求：符合知识图谱的表格形式）
    print("\n" + "=" * 50)
    with pd.ExcelWriter("古诗文知识图谱数据10.12.xlsx", engine="openpyxl") as writer:
        for table_name, table in entity_tables.items():
            table.to_excel(writer, sheet_name=table_name, index=False)

    print("所有操作完成！数据已保存至「古诗文知识图谱数据10.12.xlsx")

    # 数据概览（验证实体/关系数量）
    print("\n数据概览（文档指定实体/关系）：")
    for table_name, table in entity_tables.items():
        print(f"- {table_name}：{len(table)} 条记录")


if __name__ == "__main__":
    main()



