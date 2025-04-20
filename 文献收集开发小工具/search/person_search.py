import re
import time
from duckduckgo_search import DDGS
from types import SimpleNamespace
import requests
from bs4 import BeautifulSoup
from search.agent import BaseAgent

class WebSearch:
    def __init__(self):
        self.max_results = 10  # 增加结果数量
        self.min_content_length = 500  # 提高内容长度要求
        self.timeout = 20
        self.search_client = DDGS()
        self.personal_archive_keywords = [  # 个人档案专属关键词库
            "户籍档案", "学籍档案", "人事档案", "职称档案",
            "职业资格", "奖惩记录", "社保记录", "婚姻登记",
            "退役士兵", "党员档案", "健康档案"
        ]

    def find_links_by_query(self, query, num_results=8):
        """个人档案专用搜索优化"""
        try:
            domain_filters = "site:gov.cn OR site:12333.gov.cn OR site:moe.gov.cn"  # 人社部、教育部官网
            enhanced_query = (
                f"{query} 个人档案 "
                "管理办法 实施细则 操作指南 "
                f"({' OR '.join(self.personal_archive_keywords)}) "
                f"{domain_filters}"
            )

            results = self.search_client.text(
                keywords=enhanced_query,
                region="cn-zh",
                safesearch="strict",
                timelimit="y",
                max_results=num_results
            )

            # 个性化排序规则
            priority_rules = {
                '人社局': 5, '社会保障':4, '教育部':4, '户籍':3,
                '流动人员':5, '托管':3, '转递':2
            }
            
            processed = []
            for r in results:
                score = 0
                url = r.get('href', '')
                title = r.get('title', '')
                
                # 关键词匹配计分
                for k,v in priority_rules.items():
                    if k in title: score += v
                if '个人' in title: score += 2
                
                # 时效性判断
                if re.search(r'\d{4}年', title):
                    score += 2
                
                processed.append((-score, SimpleNamespace(
                    title=title, url=url, 
                    description=r.get('body', '')[:200]
                )))
            
            # 按分数升序排列后取前num_results
            processed.sort()
            return [x[1] for x in processed[:num_results]]

        except Exception as e:
            print(f"搜索异常: {e}")
            return []

    def fetch_page_content(self, url):
        """增强型内容获取"""
        try:
            response = requests.get(url, timeout=self.timeout)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # 政府网站特定内容定位
                main_content = soup.find(['div.mainCont', 'div.zwxxgk', 'article']) 
                if not main_content:
                    main_content = soup.body
                
                # 清理干扰元素
                for tag in ['script', 'style', 'footer', 'nav']:
                    [x.decompose() for x in main_content.find_all(tag)]
                
                text = '\n'.join([p.get_text().strip() for p in main_content.find_all(['p', 'div'])])
                return self._clean_archive_text(text)
            return ""
        except Exception as e:
            print(f"内容获取失败: {e}")
            return ""

    def _clean_archive_text(self, text):
        """个人档案专用文本清洗"""
        patterns = [
            r'（\s*更多内容.*?）',  # 移除提示语
            r'附件：.*',        # 移除附件说明
            r'咨询电话.*',       # 移除联系方式
            r'\n{3,}', '\n\n'   # 合并空行
        ]
        for p in patterns:
            text = re.sub(p, '', text)
        return text.strip()

class KnowledgeEngineerAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.knowledge_base = []
        self.personal_categories = {  # 个人档案分类体系
            '户籍类': ['出生登记', '户口迁移', '身份证办理'],
            '教育类': ['学籍档案', '学历认证', '学位证明'],
            '职业类': ['劳动合同', '职称评审', '职业资格'],
            '社会类': ['婚姻登记', '社保记录', '刑事记录']
        }

    def add_knowledge(self, content):
        """增强型知识入库"""
        try:
            pattern = r"【档案类别】(.*?)\n【信息类型】(.*?)\n【生效日期】(.*?)\n【发布单位】(.*?)\n【核心条款】(.*?)\n【原文摘要】(.*)"
            match = re.search(pattern, content, re.DOTALL)
            if match:
                entry = {
                    "档案类别": match.group(1).strip(),
                    "信息类型": self._classify_type(match.group(2)),
                    "生效日期": self._format_date(match.group(3)),
                    "发布单位": match.group(4).strip(),
                    "核心条款": self._extract_clauses(match.group(5)),
                    "原文摘要": match.group(6).strip()[:3000],
                    "入库时间": time.strftime("%Y-%m-%d %H:%M")
                }
                self.knowledge_base.append(entry)
                return f"[{entry['信息类型']}] 档案规范已收录"
            return "格式错误：请确保包含必要字段"
        except Exception as e:
            return f"解析失败：{str(e)}"

    def _classify_type(self, type_str):
        """个人档案分类"""
        for category, sub in self.personal_categories.items():
            if any(s in type_str for s in sub):
                return f"{category}-{type_str}"
        return "其他类"

class PersonalArchiveWorkflow:
    def __init__(self, research_topic):
        self.research_topic = f"个人档案专题研究：{research_topic}"
        self.engineer = KnowledgeEngineerAgent()
        self.searcher = WebSearch()
        self.max_steps = 20  # 增加研究深度

    def execute_research(self):
        feedback = ""
        for step in range(self.max_steps):
            command = self.engineer.generate_command(
                prompt=f"{self.research_topic}（需包含：{', '.join(self.searcher.personal_archive_keywords)}）",
                phase="web_research",
                step=step,
                last_feedback=feedback
            )
            
            if "```SEARCH" in command:
                query = self._extract_search_query(command)
                print(f"\033[1;36m生成检索式：{query}\033[0m")
                results = self.searcher.find_links_by_query(query)
                feedback = self._format_results(results)
                
            elif "```FETCH_DOC" in command:
                url = self._extract_url(command)
                print(f"\033[1;35m抓取文档：{url}\033[0m")
                content = self.searcher.fetch_page_content(url)
                feedback = self._analyze_content(content)
                
            elif "```ADD_KNOWLEDGE" in command:
                content = self._extract_knowledge(command)
                result = self.engineer.add_knowledge(content)
                feedback = f"\033[1;32m知识入库：{result}\033[0m"
                print(feedback)
                
            yield feedback

    def generate_full_report(self):
        """生成完整个人档案报告"""
        report = [
            "# 个人档案管理规范研究报告",
            "## 核心法规清单",
            *self._generate_legal_list(),
            "\n## 分类规范详解",
            *self._generate_category_details(),
            "\n## 业务办理指南",
            *self._generate_guidelines()
        ]
        return '\n'.join(report)

    def _generate_legal_list(self):
        """生成法规清单"""
        return [
            f"- {entry['发布单位']}《{entry['档案类别']}》 ({entry['生效日期']})"
            for entry in self.engineer.knowledge_base
        ]

if __name__ == "__main__":
    # 示例执行流程
    workflow = PersonalArchiveWorkflow("高校毕业生档案转递规范")
    
    print("开始个人档案专题研究...")
    for idx, feedback in enumerate(workflow.execute_research()):
        print(f"[步骤 {idx+1}] {feedback}")
        time.sleep(1)
    
    print("\n\033[1;31m研究成果报告：\033[0m")
    print(workflow.generate_full_report())