import os
import re
import time
import arxiv
from pypdf import PdfReader
from openai import OpenAI


class ArxivSearch:
    def __init__(self):
        # Construct the default API client
        self.sch_engine = arxiv.Client()
        self.max_query_len = 300

    def _process_query(self, query: str) -> str:
        """Process query string to fit within MAX_QUERY_LENGTH"""
        max_query_length = 300

        if len(query) <= max_query_length:
            return query

        words = query.split()
        processed_query = []
        current_length = 0

        for word in words:
            if current_length + len(word) + 1 <= max_query_length:
                processed_query.append(word)
                current_length += len(word) + 1
            else:
                break

        return ' '.join(processed_query)

    def find_papers_by_str(self, query, n=20):
        processed_query = self._process_query(query)
        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            try:
                search = arxiv.Search(
                    query="abs:" + processed_query,
                    max_results=n,
                    sort_by=arxiv.SortCriterion.Relevance
                )

                paper_sums = []
                for r in self.sch_engine.results(search):
                    paperid = r.pdf_url.split("/")[-1]
                    pubdate = str(r.published).split(" ")[0]
                    paper_sum = (
                        f"Title: {r.title}\n"
                        f"Summary: {r.summary}\n"
                        f"Publication Date: {pubdate}\n"
                        f"arXiv paper ID: {paperid}\n"
                    )
                    paper_sums.append(paper_sum)
                time.sleep(2.0)
                return "\n".join(paper_sums)

            except Exception as e:
                retry_count += 1
                if retry_count < max_retries:
                    time.sleep(2 * retry_count)
                    continue
        return None

    def retrieve_full_paper_text(self, query, max_len=50000):
        pdf_text = ""
        paper = next(arxiv.Client().results(arxiv.Search(id_list=[query])))
        paper.download_pdf(filename="downloaded-paper.pdf")

        reader = PdfReader('downloaded-paper.pdf')
        for page_number, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text()
            except Exception as e:
                os.remove("downloaded-paper.pdf")
                time.sleep(2.0)
                return "EXTRACTION FAILED"

            pdf_text += f"--- Page {page_number} ---"
            pdf_text += text + "\n"

        os.remove("downloaded-paper.pdf")
        time.sleep(2.0)
        return pdf_text[:max_len]


class BaseAgent:
    def __init__(self, model="deepseek-chat", max_steps=100, openai_api_key=None):
        self.max_steps = max_steps
        self.model = model
        self.history = []
        self.lit_review_sum = ""
        self.openai_api_key = openai_api_key
        self.prev_comm = ""
        self.max_hist_len = 10

    def set_model_backbone(self, model):
        self.model = model

    @staticmethod
    def clean_text(text):
        return text.replace("```\n", "```")

    def override_inference(self, query, temp=0.0):
        sys_prompt = f"You are {self.role_description()}"
        return query_model(
            model_str=self.model,
            system_prompt=sys_prompt,
            prompt=query,
            temp=temp,
            openai_api_key=self.openai_api_key
        )

    def inference(self, research_topic, phase, step, feedback="", temp=None):
        sys_prompt = (
            f"You are {self.role_description()}\n"
            f"Task instructions: {self.phase_prompt()}\n"
            f"{self.command_descriptions(phase)}"
        )
        history_str = "\n".join([_[1] for _ in self.history])
        complete_str = ""
        if step / (self.max_steps - 1) > 0.7:
            complete_str = "You must finish this task and submit ASAP!"

        prompt = (
            f"{'~' * 10}\nHistory: {history_str}\n{'~' * 10}\n"
            f"Current Step #{step}, Phase: {phase}\n{complete_str}\n"
            f"[Objective] Research topic: {research_topic}\n"
            f"Feedback: {feedback}\nPrevious command: {self.prev_comm}\n"
            "Please produce a single command below:\n"
        )

        model_resp = query_model(
            model_str=self.model,
            system_prompt=sys_prompt,
            prompt=prompt,
            temp=temp,
            openai_api_key=self.openai_api_key
        )
        model_resp = self.clean_text(model_resp)
        self.prev_comm = model_resp

        steps_exp = None
        if feedback and "```EXPIRATION" in feedback:
            steps_exp = int(feedback.split("\n")[0].replace("```EXPIRATION ", ""))
            feedback = extract_prompt(feedback, "EXPIRATION")

        self.history.append((
            steps_exp,
            f"Step #{step}, Phase: {phase}, Feedback: {feedback}, Response: {model_resp}"
        ))

        for _i in reversed(range(len(self.history))):
            if self.history[_i][0] is not None:
                self.history[_i] = (self.history[_i][0] - 1, self.history[_i][1])
                if self.history[_i][0] < 0:
                    self.history.pop(_i)

        if len(self.history) >= self.max_hist_len:
            self.history.pop(0)

        return model_resp

    def reset(self):
        self.history.clear()
        self.prev_comm = ""

    def role_description(self):
        raise NotImplementedError

    def phase_prompt(self):
        raise NotImplementedError

    def command_descriptions(self, phase):
        raise NotImplementedError


class PhDStudentAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.phases = ["literature review"]
        self.lit_review = []

    def requirements_txt(self):
        sys_prompt = (
            f"You are {self.role_description()}\n"
            "Task: Generate requirements.txt for the code repository"
        )
        history_str = "\n".join([_[1] for _ in self.history])
        prompt = f"History: {history_str}\n{'~' * 10}\nGenerate requirements.txt in markdown:\n"
        return query_model(
            model_str=self.model,
            system_prompt=sys_prompt,
            prompt=prompt,
            openai_api_key=self.openai_api_key
        )

    def command_descriptions(self, phase):
        return (
            "Commands:\n"
            "1. ```SUMMARY\nSEARCH_QUERY``` - Find paper summaries\n"
            "2. ```FULL_TEXT\nARXIV_ID``` - Get full paper text\n"
            "3. ```ADD_PAPER\nARXIV_ID\nSUMMARY``` - Add paper to review\n"
            "Notes:\n"
            "- Use one command per turn\n"
            "- Discuss experimental results in summaries\n"
            "- Enclose commands in triple backticks"
        )

    def phase_prompt(self):
        base_prompt = (
            "Perform literature review for the research task.\n"
            "Access arXiv to search papers and add relevant ones to review.\n"
        )
        if self.lit_review:
            papers = " ".join([paper["arxiv_id"] for paper in self.lit_review])
            return base_prompt + f"Current papers: {papers}"
        return base_prompt

    def role_description(self):
        return "a Computer Science PhD student at a top university."

    def add_review(self, review, arx_eng, agentrxiv=False, global_agentrxiv=None):
        try:
            if agentrxiv:
                arxiv_id = review.split("\n")[0]
                review_text = "\n".join(review.split("\n")[1:])
                full_text = global_agentrxiv.retrieve_full_text(arxiv_id)
            else:
                arxiv_id, review_text = review.strip().split("\n", 1)
                full_text = arx_eng.retrieve_full_paper_text(arxiv_id)

            self.lit_review.append({
                "arxiv_id": arxiv_id,
                "full_text": full_text,
                "summary": review_text,
            })
            return f"Added paper {arxiv_id}", full_text
        except Exception as e:
            return f"Error adding paper: {str(e)}", ""

    def format_review(self):
        return "Literature Review:\n" + "\n".join(
            f"ID: {p['arxiv_id']}\nSummary: {p['summary']}\n"
            for p in self.lit_review
        )


def extract_prompt(text, word):
    pattern = rf"```{word}(.*?)```"
    blocks = re.findall(pattern, text, re.DOTALL)
    return "\n".join(blocks).strip()


def query_model(
    model_str,
    prompt,
    system_prompt,
    openai_api_key=None,
    gemini_api_key=None,
    anthropic_api_key=None,
    tries=5,
    timeout=5.0,
    temp=None,
    print_cost=True,
    version="1.5"
):
    for _ in range(tries):
        if model_str == "deepseek-chat":
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            deepseek_client = OpenAI(
                api_key='sk-cfaf9101c8244f4bb274cb5ea4dbcf26',
                base_url="https://api.deepseek.com/v1"
            )
            params = {
                "model": "deepseek-chat",
                "messages": messages
            }
            if temp is not None:
                params["temperature"] = temp

            completion = deepseek_client.chat.completions.create(**params)
            return completion.choices[0].message.content


class LaboratoryWorkflow:
    def __init__(
        self,
        research_topic='',
        openai_api_key='',
        max_steps=100,
        num_papers_lit_review=3
    ):
        self.max_steps = max_steps
        self.openai_api_key = openai_api_key
        self.research_topic = research_topic
        self.num_papers_lit_review = num_papers_lit_review
        self.phd = PhDStudentAgent()
        self.arxiv_num_summaries = 5
        self.arxiv_paper_exp_time = 3
        self.reference_papers = []

    def literature_review(self):
        arx_eng = ArxivSearch()
        max_tries = self.max_steps
        resp = self.phd.inference(
            self.research_topic,
            "literature review",
            step=0,
            temp=0.4
        )

        for _i in range(max_tries):
            feedback = ""
            if "```SUMMARY" in resp:
                query = extract_prompt(resp, "SUMMARY")
                yield f"关键词:\n{query}"
                papers = arx_eng.find_papers_by_str(query, self.arxiv_num_summaries)
                yield f"摘要检索:\n{papers}"
                feedback = f"Query results for {query}:\n{papers}"

            elif "```FULL_TEXT" in resp:
                query = extract_prompt(resp, "FULL_TEXT")
                yield f"阅读论文编号:\n{query}"
                full_text = arx_eng.retrieve_full_paper_text(query)
                feedback = f"```EXPIRATION {self.arxiv_paper_exp_time}\n{full_text}```"
                yield f"论文全文:\n{feedback}"

            elif "```ADD_PAPER" in resp:
                query = extract_prompt(resp, "ADD_PAPER")
                feedback, text = self.phd.add_review(query, arx_eng)
                if len(self.reference_papers) < self.num_papers_lit_review:
                    self.reference_papers.append(text)
                yield f"知识入库:\n{query}"

            if len(self.phd.lit_review) >= self.num_papers_lit_review:
                lit_review_sum = self.phd.format_review()
                yield f"知识预览:\n{lit_review_sum}"
                self.phd.lit_review = []
                return False

            resp = self.phd.inference(
                self.research_topic,
                "literature review",
                feedback=feedback,
                step=_i + 1,
                temp=0.4
            )


if __name__ == '__main__':
    lab = LaboratoryWorkflow(research_topic="数字化经济", openai_api_key="")
    for chunk in lab.literature_review():
        print(chunk)