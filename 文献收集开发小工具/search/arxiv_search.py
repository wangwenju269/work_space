import os
import re
import time
import json
import arxiv
from typing import List, Dict, Generator, Optional, Tuple
from pypdf import PdfReader
from search.agent import BaseAgent

class ArxivEngineer:
    """Arxiv论文搜索工具类"""
    MAX_QUERY_LENGTH = 300
    DOWNLOAD_FILENAME = "downloaded-paper.pdf"
    
    def __init__(self):
        self.client = arxiv.Client()
        self.max_retries = 3
        self.retry_delay = 2.0

    def _truncate_query(self, query: str) -> str:
        """截断查询字符串以适应最大长度"""
        if len(query) <= self.MAX_QUERY_LENGTH:
            return query

        truncated = []
        current_length = 0
        for word in query.split():
            if current_length + len(word) + 1 > self.MAX_QUERY_LENGTH:
                break
            truncated.append(word)
            current_length += len(word) + 1
        return " ".join(truncated)

    def search_papers(self, query: str, max_results: int = 20) -> Optional[str]:
        """搜索论文摘要"""
        query = self._truncate_query(query)
        search = arxiv.Search(
            query=f"abs:{query}",
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )

        for retry in range(self.max_retries):
            try:
                results = []
                for result in self.client.results(search):
                    paper_id = result.pdf_url.split("/")[-1]
                    pub_date = str(result.published).split(" ")[0]
                    results.append(
                        f"Title: {result.title}\n"
                        f"Summary: {result.summary}\n"
                        f"Publication Date: {pub_date}\n"
                        f"arXiv paper ID: {paper_id}\n"
                    )
                time.sleep(self.retry_delay)
                return "\n".join(results)
            except Exception as e:
                if retry < self.max_retries - 1:
                    time.sleep(self.retry_delay * (retry + 1))
        return None

    def get_full_text(self, paper_id: str, max_length: int = 50000) -> str:
        """获取论文全文"""
        try:
            paper = next(self.client.results(arxiv.Search(id_list=[paper_id])))
            paper.download_pdf(filename=self.DOWNLOAD_FILENAME)

            text = []
            reader = PdfReader(self.DOWNLOAD_FILENAME)
            for page_num, page in enumerate(reader.pages, 1):
                try:
                    page_text = page.extract_text()
                    text.append(f"--- Page {page_num} ---\n{page_text}")
                except Exception as e:
                    return "EXTRACTION_FAILED"
                
            full_text = "\n".join(text)[:max_length]
            return full_text
        finally:
            if os.path.exists(self.DOWNLOAD_FILENAME):
                os.remove(self.DOWNLOAD_FILENAME)
            time.sleep(self.retry_delay)


class AcademicAgent(BaseAgent):
    """博士生研究助手"""
    PHASES = ["literature_review"]
    
    def __init__(self):
        super().__init__()
        self.literature_reviews: List[Dict] = []
        self.downloaded_papers: Dict[str, str] = {}  # 新增：存储已下载的全文

    def role_description(self) -> str:
        return "a student at a top university."

    def command_descriptions(self) -> str:
        return (
            "Available Commands:\n"
            "1. Search paper summaries:\n"
            "```SUMMARY\n<search_query>\n```\n"
            "2. Retrieve full paper text:\n"
            "```FULL_TEXT\n<arxiv_id>\n```\n"
            "3. Add paper to review:\n"
            "```ADD_PAPER\n<arxiv_id>\n<summary>\n```\n"
            "Notes:\n"
            "- ADD_PAPER must be preceded by FULL_TEXT command (cannot follow SUMMARY)\n"
            "- Avoid excessive SUMMARY commands\n"
            "- Keep search querie concise\n"
            "- Only one command per response\n"
            "- Always wrap commands in triple backticks"
        )

    def phase_prompt(self) -> str:
        review_status = (
            "Papers in review: " + 
            ", ".join(p["arxiv_id"] for p in self.literature_reviews)
            if self.literature_reviews else ""
        )
        return (
            "Perform comprehensive literature review for the research topic.\n"
            "Access arXiv to search and evaluate relevant papers.\n"
            f"{review_status}"
        )

    def add_literature_reviews(self, content: str) -> str:
        try:
            # 解析输入内容
            parts = content.split("\n", 1)
            if len(parts) < 2:
                return "Invalid format. Expected: arxiv_id\\nsummary"
            
            arxiv_id, summary = parts[0].strip(), parts[1].strip()
            
            if arxiv_id not in self.downloaded_papers:
                return f"Full text for {arxiv_id} not found. Use FULL_TEXT command first."
            
            self.literature_reviews.append(
                {
                    "arxiv_id": arxiv_id,
                    "summary": summary,
                    "full_text": self.downloaded_papers[arxiv_id]
                }
            )
            return f"Successfully added paper {arxiv_id}"
        
        except Exception as e:
            return f"Error adding paper: {str(e)}"     
        
    def generate_review_report(self, research_topic: str) -> str:
        """生成文献综述报告"""
        if not self.literature_reviews:
            return "尚无文献可供分析"
        
        # 构建结构化提示词
        system_prompt = (
            f"基于以下研究论文为【{research_topic}】生成结构化文献综述：\n"
            "要求：\n"
            "1. 按逻辑组织成章节，包含：引言、方法总结、主要发现、研究比较、不足与未来方向\n"
            "2. 突出各研究的创新点和技术路线\n"
            "3. 对比分析不同方法的优缺点\n"
            "4. 指出当前研究空白\n"
            "5. 使用学术写作风格，包含参考文献标注\n"
        )
        
        # 添加论文上下文
        prompt = str()
        for idx, paper in enumerate(self.literature_reviews, 1):
            prompt += (
                f"\n=== 论文 {idx} [{paper['arxiv_id']}] ===\n"
                f"标题：{paper.get('title', '')}\n"
                f"摘要：{paper['summary'][:1000]}...\n"
                f"关键内容：{self._extract_key_points(paper['full_text'])}\n"
            )
        
        # 生成报告
        return self._query_model(
            system_prompt=system_prompt,
            prompt=prompt,
            temperature=0.8,
            max_tokens=2500
        )
        
    def _extract_key_points(self, text: str) -> str:
        """从全文提取关键点（示例实现）"""
        keywords = ["method", "propose", "result", "experiment", "contrast"]
        sentences = []
        for sent in re.split(r'(?<=[.!?]) +', text):
            if any(kw in sent.lower() for kw in keywords):
                sentences.append(sent.strip())
        return " ".join(sentences)    
        

class AcademicWorkflow:

    def __init__(
        self,
        max_steps: int = 100,
        target_papers: int = 5
    ):
        self.agent = AcademicAgent()
        self.engine = ArxivEngineer()
        self.max_steps = max_steps
        self.target_papers = target_papers
        self.research_topic = None
 
    def run(self, research_topic: str) -> Generator:
        """主运行流程"""
        self.research_topic = research_topic
        self.agent.literature_reviews = []
        return self._literature_review_phase()
        
    def _literature_review_phase(self) -> Generator[str, None, None]:
        """文献综述阶段生成器"""
        feedback = str()
        for step in range(self.max_steps):
            # 生成命令
            command = self.agent.generate_command(
                research_topic=self.research_topic,
                phase="literature_review",
                step=step,
                feedback = feedback,
                temperature=0.9
            )

            # 处理命令
            if "```SUMMARY" in command:
                query = self.agent.extract_prompt(command, "SUMMARY")
                print(f"学术搜索:\n{query}")
            
                papers = self.engine.search_papers(query)
                feedback = f"Found papers for: {query}\n{papers}" if papers else "No found papers"
                
                data_json = {
                        'type': 'SUMMARY',
                        'query': query,
                        'feedback': feedback
                }   
                
            elif "```FULL_TEXT" in command:
                paper_id = self.agent.extract_prompt(command, "FULL_TEXT").strip()
                print(f"获取论文:\n{paper_id}")
                
                full_text = self.engine.get_full_text(paper_id)
                if full_text:
                    self.agent.downloaded_papers[paper_id] = full_text
                    feedback = f"```{full_text[:1000]}...```"  # 显示前1000字符
                else:
                    feedback = "内容获取失败"
                
                data_json = {
                    'type': 'FULL_TEXT',
                    'paper_id': paper_id,
                    'feedback': feedback
                }
                
            elif "```ADD_PAPER" in command:
                content  = self.agent.extract_prompt(command, "ADD_PAPER")  
                print(f"添加文献：{content[:500]}...")  
                
                result = self.agent.add_literature_reviews(content)
                feedback = f"文献添加结果：{result}" 
                 
                data_json = {
                        'type': 'ADD_PAPER',
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
            
            # completion condition
            if  len(self.agent.literature_reviews) >= self.target_papers:
                return
     
    def generate_llm_report(self) -> str:
        """生成最终研究报告"""
        if not self.agent.literature_reviews:
            return "尚未收集到任何文献"
        
        report = [
            "# 文献综述报告",
            f"**研究主题**: {self.research_topic}",
            f"**分析文献数**: {len(self.agent.literature_reviews)}",
        ]
        
        # 生成AI综述
        ai_review = self.agent.generate_review_report(self.research_topic)
        report.append(ai_review)
        
        # 添加参考文献列表
        report.append("\n## 附录")
        for paper in self.agent.literature_reviews:
            report.append(
                f"- [{paper['arxiv_id']}] {paper.get('title', '')}\n"
                f"  摘要: {paper['summary'][:200]}..."
            )
        return "\n".join(report)        
                      
if __name__ == "__main__":
    workflow = AcademicWorkflow(
        max_steps=10,
        target_papers=3
    )
    
    print("启动学术研究流程...")
    for feedback in workflow.run("大语言模型推理优化"):
        print(f"当前进度：\n{json.dumps(feedback, ensure_ascii=False)}\n{'='*50}")
    
    print("\n最终研究报告：")
    print(workflow.generate_llm_report())     