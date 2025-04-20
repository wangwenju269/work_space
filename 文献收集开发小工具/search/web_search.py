import time
import json
from duckduckgo_search import DDGS
from types import SimpleNamespace
import requests
from bs4 import BeautifulSoup
from search.agent import BaseAgent

Color = lambda text: f'\033[48;5;9m\033[5m{text}\033[0m'

class WebSearch:
    def __init__(self):
        self.max_results = 10
        self.min_content_length = 500
        self.timeout = 10
        self.search_client = DDGS()  # 初始化DuckDuckGo客户端
        
    def _clean_html(self, html):
        """使用BeautifulSoup清理网页内容"""
        soup = BeautifulSoup(html, 'html.parser')
        # 移除无关元素
        for element in soup(['script', 'style', 'header', 'footer', 'nav']):
            element.decompose()
        return soup.get_text(separator='\n', strip=True)

    def find_links_by_query(self, query, num_results=5):
        """使用DuckDuckGo搜索获取相关链接"""
        try:
            results = self.search_client.text(
                keywords=query,
                region="cn-zh",  # 中国区中文结果
                safesearch="moderate",  # 安全搜索级别
                timelimit="y",  # 最近一年
                max_results=num_results
            )
            
            # 转换为与原来兼容的格式
            return [SimpleNamespace(
                title=result.get('title', ''),
                url=result.get('href', ''),
                description=result.get('body', '')
            ) for result in results]
            
        except Exception as e:
            print(f"DuckDuckGo搜索失败: {str(e)}")
            return []


    def fetch_page_content(self, url):
        """获取并清理网页内容，支持重试和编码处理"""
        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=self.timeout)
                if response.status_code == 200:
                    # 编码处理流程
                    if response.encoding == 'ISO-8859-1':
                        # 尝试检测实际编码
                        content = response.content
                        for encoding in ['utf-8', 'gbk', 'gb2312']:
                            try:
                                text = content.decode(encoding)
                                response.encoding = encoding
                                break
                            except UnicodeDecodeError:
                                continue
                        else:
                            text = content.decode('utf-8', errors='replace')

                    cleaned = self._clean_html(response.text)
                    return cleaned[:20000]
                
            except (requests.exceptions.RequestException, UnicodeDecodeError) as e:
                print(f"尝试 {attempt+1}/{max_retries} 失败: {str(e)}")
                if attempt < max_retries - 1:
                    sleep_time = (attempt + 1) * 2  # 指数退避
                    print(f"{sleep_time}秒后重试...")
                    time.sleep(sleep_time)
                continue
                
        print(f"全部 {max_retries} 次尝试均失败")
        return ""


class KnowledgeEngineerAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.phases = ["web_research"]
        self.knowledge_base = []

    def command_descriptions(self):
        return (
            "生成搜索关键词: ```SEARCH\n查询简短关键词\n```\n"
            "获取网页内容: ```FETCH_DOC\n目标URL\n```\n"
            "添加知识条目: ```ADD_KNOWLEDGE\n知识摘要\n相关证据\n```\n"
            "确保每个步骤只执行一个命令，用```包裹命令"
        )

    def phase_prompt(self):
        return (
            f"当前知识条目数量: {len(self.knowledge_base)}\n"
            "通过网页调研收集可靠信息，重点关注：\n"
            "- 权威机构的报告\n- 学术论文\n- 行业白皮书\n- 统计数据\n"
            "避免论坛帖子、广告等低质量来源"
        )

    def add_knowledge(self, content):
        """添加结构化知识条目"""
        try:
            parts = content.split("\n", 1)
            if len(parts) == 2:
                summary, evidence = parts
                self.knowledge_base.append({
                    "summary": summary.strip(),
                    "evidence": evidence.strip()[:5000],
                    "timestamp": time.strftime("%Y-%m-%d")
                })
                return "知识添加成功"
            return "格式错误：需要摘要和证据两部分"
        except Exception as e:
            return f"处理失败：{str(e)}"

class ResearchWorkflow:
    def __init__(self,max_steps=20):
        self.engineer = KnowledgeEngineerAgent()
        self.searcher = WebSearch()
        self.max_steps = max_steps
        self.research_topic = None  # 延迟初始化

    def run(self, research_topic):
        """主运行入口"""
        self.research_topic = research_topic
        self.engineer.knowledge_base = []  # 重置知识库
        return self._execute_web_research()
    
    def _execute_web_research(self):
        feedback = ""
        for step in range(self.max_steps):
            # 生成搜索命令
            command = self.engineer.generate_command(self.research_topic, "web_research", step, feedback)
            
            if "```SEARCH" in command:
                query = self.engineer.extract_prompt(command, "SEARCH")
                print(Color(f"生成搜索关键词: {query}"))
                
                # 执行网页搜索
                results = self.searcher.find_links_by_query(query)
                # 修改点：增加搜索结果判断
                if results:
                    feedback = "找到以下资源：\n" + "\n".join([f"{r.title}: {r.url}" for r in results])
                else:
                    feedback = "未找到相关资源，建议尝试以下方法：\n1. 使用更精确的关键词\n2. 添加限定词（如最新年份）\n3. 换用相关术语"
                
                data_json = {
                        'type': 'SEARCH',
                        'query': query,
                        'feedback': feedback
                }
                
            elif "```FETCH_DOC" in command:
                url = self.engineer.extract_prompt(command, "FETCH_DOC")
                print(Color(f"抓取网页内容: {url}"))
                
                # 获取并清洗内容
                content = self.searcher.fetch_page_content(url)
                feedback = f"网页内容摘要：\n{content[:1000]}..." if content else "内容获取失败，可能原因：\n- 页面加载超时\n- 需要验证访问\n- 网站反爬机制"
                
                data_json = {
                    'type': 'FETCH_DOC',
                    'url': url,
                    'feedback': feedback
                }
                
            elif "```ADD_KNOWLEDGE" in command:
                content = self.engineer.extract_prompt(command, "ADD_KNOWLEDGE")
                print(Color(f"添加知识条目: {content}..."))
                
                # 结构化存储知识
                result = self.engineer.add_knowledge(content)
                feedback = f"知识存储结果：{result}"
                
                data_json = {
                        'type': 'ADD_KNOWLEDGE',
                        'content': "添加知识",
                        'feedback': content
                }
                
            else:
                feedback = "无效命令格式，请确认使用正确命令语法"
                data_json = {
                        'type': 'error',
                        'content': "执行错误",
                        'feedback': feedback
                }
            yield data_json
        
    def format_report(self):
        """生成结构化研究报告"""
        report = ["# 研究主题分析报告", f"## {self.research_topic}"]
        for idx, entry in enumerate(self.engineer.knowledge_base, 1):
            report.append(f"### 发现 {idx}\n**摘要**: {entry['summary']}\n\n**证据**: {entry['evidence']}")
        return "\n".join(report)
    
    # 在ResearchWorkflow类中添加以下方法
    def generate_llm_report(self):
        inference = self.format_report()
        
        # 构建提示词
        system_prompt = f"""你是一位资深行业分析师，需要根据以下调研数据生成专业研究报告。"""
        prompt= f"""调研主题：{self.research_topic}\n调研数据：\n{inference}"""
       
        return self.engineer._query_model(
            system_prompt=system_prompt,
            prompt=prompt
        )
    
    
# 使用示例
if __name__ == "__main__":
    workflow = ResearchWorkflow(
        max_steps=10
    )
    
    # 执行调研流程
    for feedback in workflow.run("绿色能源发展趋势"):
        print(f"当前进度：\n{json.dumps(feedback, ensure_ascii=False)}\n{'='*50}")
    
    # 生成报告
    print("\nMarkdown格式报告：")
    print(workflow.format_report())
    
    print("\nLLM生成报告：")
    print(workflow.generate_llm_report("llm"))