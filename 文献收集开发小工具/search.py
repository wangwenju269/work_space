import os
import re
import time
from typing import List, Dict, Generator, Optional, Tuple

import arxiv
from pypdf import PdfReader
from openai import OpenAI


class ArxivSearch:
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


class BaseAgent:
    """基础智能体类"""
    MAX_HISTORY_LENGTH = 100
    
    def __init__(
        self,
        model: str = "deepseek-chat",
        max_steps: int = 100,
        api_key: Optional[str] = None,
        max_hist_len: int = 6
    ):
        self.model = model
        self.max_steps = max_steps
        self.history: List[Tuple[Optional[int], str]] = []
        self.api_key = api_key
        self.previous_command = ""
        self.max_hist_len = max_hist_len

    def reset(self):
        """重置对话历史"""
        self.history.clear()
        self.previous_command = ""

    @staticmethod
    def clean_response(text: str) -> str:
        """清理模型响应"""
        return text.replace("```\n", "```")

    def role_description(self) -> str:
        """角色描述"""
        return "a helpful AI assistant"

    def generate_command(
        self,
        research_topic: str,
        phase: str,
        step: int,
        feedback: str = "",
        temperature: Optional[float] = None
    ) -> str:
        """生成命令"""
        system_prompt = (
            f"You are {self.role_description()}\n"
            f"Task instructions: {self.phase_prompt()}\n"
            f"{self.command_descriptions()}"
        )

        history_str = "\n".join(entry[1] for entry in self.history)
        complete_note = "You must finish this task and submit as soon as possible!" if step/(self.max_steps-1) > 0.7 else ""

        prompt = (
            f"{'~'*10} History {'~'*10}\n{history_str}\n{'~'*28}\n"
            f"Current Step #{step}, Phase: {phase}\n{complete_note}\n"
            f"[Research Topic] {research_topic}\n"
            f"Feedback: {feedback}\n"
            f"Previous command: {self.previous_command}\n"
            "Please produce a single command below:\n"
        )

        response = self._query_model(
            system_prompt=system_prompt,
            prompt=prompt,
            temperature=temperature
        )
        self.previous_command = response
        steps_exp = None
        if feedback is not None and "```EXPIRATION" in feedback:
            steps_exp = int(feedback.split("\n")[0].replace("```EXPIRATION ", ""))
            feedback = self.extract_prompt(feedback, "EXPIRATION")
            
        self.history.append((steps_exp, f"Step #{step}, Phase: {phase}, Feedback: {feedback}, Your response: {response}"))
        for _i in reversed(range(len(self.history))):
            if self.history[_i][0] is not None:
                self.history[_i] = (self.history[_i][0] - 1, self.history[_i][1])
                if self.history[_i][0] < 0:
                    self.history.pop(_i)
        if len(self.history) >= self.max_hist_len:
            self.history.pop(0)
        
        return self.clean_response(response) 
        
    def extract_prompt(
        self,
        text: str,
        word: str
    ) -> str:
        code_block_pattern = rf"```{word}(.*?)```"
        code_blocks = re.findall(code_block_pattern, text, re.DOTALL)
        extracted_code = "\n".join(code_blocks).strip()
        return extracted_code  

    def _query_model(
        self,
        system_prompt: str,
        prompt: str,
        temperature: Optional[float] = None
    ) -> str:
        """调用模型API"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        client = OpenAI(
            api_key=self.api_key or "sk-cfaf9101c8244f4bb274cb5ea4dbcf26",
            base_url="https://api.deepseek.com/v1"
        )

        params = {"model": "deepseek-chat", "messages": messages}
        if temperature is not None:
            params["temperature"] = temperature

        response = client.chat.completions.create(**params)
        return response.choices[0].message.content


class PhDStudentAgent(BaseAgent):
    """博士生研究助手"""
    PHASES = ["literature_review"]
    
    def __init__(self):
        super().__init__()
        self.literature_reviews: List[Dict] = []

    def role_description(self) -> str:
        return "a computer science PhD student at a top university."

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
            "- Keep search queries concise\n"
            "- Only one command per response\n"
            "- Always wrap commands in triple backticks"
            "- Make sure to use ADD_PAPER when you see a relevant paper. DO NOT use SUMMARY too many times."
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

    def add_paper_review(
        self,
        arxiv_id: str,
        summary: str,
        full_text: str
    ) -> None:
        """添加论文到文献综述"""
        self.literature_reviews.append({
            "arxiv_id": arxiv_id,
            "summary": summary,
            "full_text": full_text
        })

    def format_review(self) -> str:
        """生成文献综述报告"""
        return "\n".join(
            f"arXiv ID: {p['arxiv_id']}\nSummary: {p['summary']}\n"
            for p in self.literature_reviews
        )


class ResearchWorkflow:
    """研究工作流管理器"""
    def __init__(
        self,
        research_topic: str,
        api_key: Optional[str] = None,
        max_steps: int = 100,
        target_papers: int = 3
    ):
        self.research_topic = research_topic
        self.api_key = api_key
        self.max_steps = max_steps
        self.target_papers = target_papers
        self.assistant = PhDStudentAgent()
        self.arxiv_tool = ArxivSearch()

    def literature_review_phase(self) -> Generator[str, None, None]:
        """文献综述阶段生成器"""
        feedback = str()
        for step in range(self.max_steps):
            # 生成命令
            command = self.assistant.generate_command(
                research_topic=self.research_topic,
                phase="literature_review",
                step=step,
                feedback = feedback,
                temperature=0.4
            )

            # 处理命令
            if "```SUMMARY" in command:
                query = self._extract_command_content(command, "SUMMARY")
                yield f"关键词检索:\n{query}"
                
                papers = self.arxiv_tool.search_papers(query)
                yield f"摘要结果:\n{papers}"
                
                feedback = f"Found papers for: {query}\n{papers}"
                
            elif "```FULL_TEXT" in command:
                paper_id = self._extract_command_content(command, "FULL_TEXT")
                yield f"获取全文:\n{paper_id}"
                
                full_text = self.arxiv_tool.get_full_text(paper_id)
                feedback = f"```EXPIRATION 5\n{full_text}```"
                yield f"论文内容:\n{full_text[:500]}..."
                
            elif "```ADD_PAPER" in command:
                content = self._extract_command_content(command, "ADD_PAPER")
                paper_id, summary = content.split("\n", 1)
                full_text = self.arxiv_tool.get_full_text(paper_id)
                
                self.assistant.add_paper_review(paper_id, summary, full_text)
                feedback = f"Added paper {paper_id}"
                yield f"论文入库:\n{paper_id}"
                
            # 检查完成条件
            if len(self.assistant.literature_reviews) >= self.target_papers:
                report = self.assistant.format_review()
                yield f"文献综述完成:\n{report}"
                return

    @staticmethod
    def _extract_command_content(text: str, command: str) -> str:
        """从命令文本中提取内容"""
        pattern = rf"```{command}\n(.*?)```"
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else ""


if __name__ == "__main__":
    workflow = ResearchWorkflow(
        research_topic="数字化经济",
        api_key="your_api_key_here"
    )
    
    for output in workflow.literature_review_phase():
        print(output)
        print("\n" + "="*50 + "\n")