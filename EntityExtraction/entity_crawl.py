import os
import re
import requests
from bs4 import BeautifulSoup
import time
import urllib.parse

# 基础配置
BASE_URL = "https://www.gushiwen.cn"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Referer": "https://www.gushiwen.cn/"
}
# 爬取"词"这种文体形式
STYLE_NAME = "词"
STYLE_URL = "/shiwens/default.aspx?"
TARGET_COUNT = 300  # 目标爬取数量
OUTPUT_FILE = "../CrawlToTextLine.txt"


def fetch_page(url: str) -> BeautifulSoup:
    """爬虫页面请求（反爬延时+异常捕获）"""
    try:
        time.sleep(1)  # 避免高频请求被封
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        return BeautifulSoup(response.text, "html.parser")
    except Exception as e:
        print(f"页面请求失败：{url}，错误：{str(e)}")
        return None


def extract_poem_info(poem_url: str) -> dict:
    """提取单篇诗文信息：标题、作者、朝代、内容"""
    soup = fetch_page(poem_url)
    if not soup:
        return None

    # 定位诗文核心容器
    poem_container = soup.find("div", id=lambda x: x and x.startswith("zhengwen"))
    if not poem_container:
        print(f"单篇诗文页[{poem_url}]未找到核心容器")
        return None

    # 提取标题
    title_tag = poem_container.find("h1")
    title = title_tag.text.strip() if (title_tag and title_tag.text.strip()) else "未知标题"

    # 提取作者和朝代
    author = "未知作者"
    dynasty = "未知朝代"
    source_tag = poem_container.find("p", class_="source")
    if source_tag:
        author_tag = source_tag.find("a", href=lambda x: x and "/authorv_" in x)
        if author_tag:
            author = author_tag.text.strip()

        dynasty_tag = source_tag.find("a", href=lambda x: x and "/shiwens/default.aspx?cstr=" in x)
        if dynasty_tag:
            dynasty = dynasty_tag.text.strip().strip("〔〕")

    # 提取原文内容
    content = "原文缺失"
    content_tag = poem_container.find("div", class_="contson")
    if content_tag:
        # 处理换行符
        for br in content_tag.find_all("br"):
            br.replace_with("\n")
        content = content_tag.text.strip().replace("\n", " ").replace("  ", " ")  # 替换换行为空格

    return {
        "title": title,
        "author": author,
        "dynasty": dynasty,
        "content": content
    }


def crawl_ci_poetry():
    """爬取所有词形式的诗文，至少300首"""
    crawled_poem_links = set()  # 去重集合
    current_page = 1

    print(f"开始爬取[{STYLE_NAME}]形式的诗文，目标{TARGET_COUNT}首...")

    # 循环收集足够的诗文链接
    while len(crawled_poem_links) < TARGET_COUNT:
        # 根据提供的URL修正：使用xstr参数筛选"词"这种文体
        encoded_style = urllib.parse.quote(STYLE_NAME)
        if current_page == 1:
            page_url = f"{BASE_URL}{STYLE_URL}xstr={encoded_style}"
        else:
            page_url = f"{BASE_URL}{STYLE_URL}page={current_page}&xstr={encoded_style}"

        print(f"爬取第{current_page}页（已收集{len(crawled_poem_links)}/{TARGET_COUNT}）：{page_url}")
        soup = fetch_page(page_url)
        if not soup:
            current_page += 1
            continue

        # 提取当前页的诗文链接
        poem_links = soup.find_all("a", href=lambda x: x and "/shiwenv_" in x)
        if not poem_links:
            print(f"第{current_page}页未找到诗文链接，停止爬取")
            break

        # 收集链接
        for link in poem_links:
            poem_url = f"{BASE_URL}{link['href']}"
            if poem_url not in crawled_poem_links:
                crawled_poem_links.add(poem_url)
                # 达到目标数量后停止
                if len(crawled_poem_links) >= TARGET_COUNT:
                    print(f"已收集够{TARGET_COUNT}个诗文链接，停止分页爬取")
                    break

        current_page += 1

    # 处理收集到的链接并写入文件
    print(f"开始处理{len(crawled_poem_links)}个诗文链接...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        count = 0
        for poem_url in crawled_poem_links:
            count += 1
            print(f"处理第{count}/{len(crawled_poem_links)}个：{poem_url}")
            poem_info = extract_poem_info(poem_url)
            if poem_info:
                line = f"{poem_info['title']}{poem_info['author']}{poem_info['dynasty']}{poem_info['content']}\n"
                f.write(line)
            time.sleep(0.5)  # 降低请求频率

    print(f"爬取完成，共处理{count}首，结果已写入{OUTPUT_FILE}")


def remove_brackets_content():
    if not os.path.exists(OUTPUT_FILE):
        print(f"文件{OUTPUT_FILE}不存在，无法进行处理")
        return
    # 正则表达式匹配中英文括号及其内容
    pattern = re.compile(r'[（(].*?[）)]')

    processed_lines = []

    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            cleaned_content = pattern.sub('', line)
            # 处理可能产生的多余空格
            cleaned_content = re.sub(r'  +', ' ', cleaned_content).strip()
            processed_lines.append(cleaned_content)

    # 将处理后的内容写回文件
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.writelines(processed_lines)

    print(f"已完成对{OUTPUT_FILE}的括号内容处理")

def remove_all_spaces():
    """去除文件中所有空格"""
    if not os.path.exists(OUTPUT_FILE):
        print(f"文件{OUTPUT_FILE}不存在，无法进行处理")
        return

    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 去除每行所有空格
    lines_no_space = [line.replace(" ", "") for line in lines]

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.writelines(lines_no_space)

    print(f"已去除{OUTPUT_FILE}中的所有空格")


if __name__ == "__main__":
    # crawl_ci_poetry()
    remove_brackets_content()
    remove_all_spaces()
