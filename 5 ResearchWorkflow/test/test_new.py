import os
import re
import time
import arxiv
from pypdf import PdfReader
from openai import OpenAI


class ArxivSearch:
    def __init__(self):
        self.sch_engine = arxiv.Client()
        self.max_query_len = 300

    def _process_query(self, query: str) -> str:
        MAX_QUERY_LENGTH = 300
        if len(query) <= MAX_QUERY_LENGTH:
            return query

        words = query.split()
        processed_query = []
        current_length = 0

        for word in words:
            if current_length + len(word) + 1 <= MAX_QUERY_LENGTH:
                processed_query.append(word)
                current_length += len(word) + 1
            else:
                break

        return ' '.join(processed_query)

    def find_papers_by_str(self, query, N=20):
        processed_query = self._process_query(query)
        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            try:
                search = arxiv.Search(
                    query="abs:" + processed_query,
                    max_results=N,
                    sort_by=arxiv.SortCriterion.Relevance
                )

                paper_sums = []
                for r in self.sch_engine.results(search):
                    paperid = r.pdf_url.split("/")[-1]
                    pubdate = str(r.published).split(" ")[0]
                    paper_sum = f"Title: {r.title}\n"
                    paper_sum += f"Summary: {r.summary}\n"
                    paper_sum += f"Publication Date: {pubdate}\n"
                    paper_sum += f"arXiv paper ID: {paperid}\n"
                    paper_sums.append(paper_sum)
                time.sleep(2.0)
                return "\n".join(paper_sums)

            except Exception as e:
                retry_count += 1
                if retry_count < max_retries:
                    time.sleep(2 * retry_count)
                    continue
        return None

    def retrieve_full_paper_text(self, query, MAX_LEN=50000):
        pdf_text = str()
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
            pdf_text += text
            pdf_text += "\n"

        os.remove("downloaded-paper.pdf")
        time.sleep(2.0)
        return pdf_text[:MAX_LEN]


class BaseAgent:
    def __init__(self, model="deepseek-chat", max_steps=100, openai_api_key=None):
        self.max_steps = max_steps
        self.model = model
        self.history = list()
        self.lit_review_sum = str()
        self.openai_api_key = openai_api_key

    def set_model_backbone(self, model):
        self.model = model

    @staticmethod
    def clean_text(text):
        return text.replace("```\n", "```")

    def override_inference(self, query, temp=0.0):
        sys_prompt = f"""You are {self.role_description()}"""
        model_resp = query_model(
            model_str=self.model,
            system_prompt=sys_prompt,
            prompt=query,
            temp=temp,
            openai_api_key=self.openai_api_key
        )
        return model_resp

    def inference(self, research_topic, phase, step, feedback="", temp=None):
        sys_prompt = (
            f"You are {self.role_description()} \n"
            f"Task instructions: {self.phase_prompt()}\n"
            f"{self.command_descriptions(phase)}"
        )
        history_str = "\n".join([_[1] for _ in self.history])
        complete_str = "You must finish this task and submit as soon as possible!" if step/(self.max_steps-1) > 0.7 else ""
        
        prompt = (
            f"{'~' * 10}\nHistory: {history_str}\n{'~' * 10}\n"
            f"Current Step #{step}, Phase: {phase}\n{complete_str}\n"
            f"[Objective] Your goal is to perform research on: {research_topic}\n"
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


class PhDStudentAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.phases = ["literature review"]
        self.lit_review = []

    def requirements_txt(self):
        sys_prompt = (
            f"You are {self.role_description()} \n"
            "Task instructions: Generate a requirements.txt for the github repository."
        )
        history_str = "\n".join([_[1] for _ in self.history])
        prompt = f"History: {history_str}\n{'~' * 10}\nPlease produce requirements.txt:\n"
        
        model_resp = query_model(
            model_str=self.model,
            system_prompt=sys_prompt,
            prompt=prompt,
            openai_api_key=self.openai_api_key
        )
        return model_resp

    def command_descriptions(self):
        return (
            "To collect paper summaries: ```SUMMARY\nSEARCH QUERY```\n"
            "To get full paper text: ```FULL_TEXT\narXiv_ID```\n"
            "To add paper: ```ADD_PAPER\narXiv_ID\nSUMMARY```\n"
            "Use one command per turn. Include ```COMMAND\ntext``` format."
        )

    def phase_prompt(self):
        phase_str = (
            "Perform literature review for the research task.\n"
            "Access arXiv to search papers and add relevant ones.\n"
        )
        rev_papers = "Current papers: " + " ".join([_paper["arxiv_id"] for _paper in self.lit_review]) if self.lit_review else ""
        return phase_str + rev_papers

    def role_description(self):
        return "a computer science PhD student at a top university."

    def add_review(self, review, arx_eng, agentrxiv=False, GLOBAL_AGENTRXIV=None):
        try:
            if agentrxiv:
                arxiv_id = review.split("\n")[0]
                review_text = "\n".join(review.split("\n")[1:])
                full_text = GLOBAL_AGENTRXIV.retrieve_full_text(arxiv_id)
            else:
                arxiv_id, review_text = review.strip().split("\n", 1)
                full_text = arx_eng.retrieve_full_paper_text(arxiv_id)

            review_entry = {
                "arxiv_id": arxiv_id,
                "full_text": full_text,
                "summary": review_text,
            }
            self.lit_review.append(review_entry)
            return f"Added paper {arxiv_id}", full_text
        except Exception as e:
            return f"Error adding review: {str(e)}", ""

    def format_review(self):
        return "Literature Review:\n" + "\n".join(
            f"arXiv ID: {_l['arxiv_id']}, Summary: {_l['summary']}"
            for _l in self.lit_review
        )


def extract_prompt(text, word):
    code_block_pattern = rf"```{word}(.*?)```"
    code_blocks = re.findall(code_block_pattern, text, re.DOTALL)
    return "\n".join(code_blocks).strip()


def query_model(
    model_str, prompt, system_prompt, openai_api_key=None, gemini_api_key=None,
    anthropic_api_key=None, tries=5, timeout=5.0, temp=None, print_cost=True, version="1.5"
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
        self, research_topic='', openai_api_key='',
        max_steps=100, num_papers_lit_review=3
    ):
        self.max_steps = max_steps
        self.openai_api_key = openai_api_key
        self.research_topic = research_topic
        self.num_papers_lit_review = num_papers_lit_review
        self.phd = PhDStudentAgent()

    def literature_review(self):
        arx_eng = ArxivSearch()
        max_tries = self.max_steps
        resp = self.phd.inference(self.research_topic, "literature review", step=0, temp=0.4)

        for _i in range(max_tries):
            feedback = str()
            if "```SUMMARY" in resp:
                query = extract_prompt(resp, "SUMMARY")
                yield f"关键词:\n{query}"
                papers = arx_eng.find_papers_by_str(query, N=self.arxiv_num_summaries)
                yield f"摘要检索:\n{papers}"
                feedback = f"arXiv papers for {query}:\n{papers}"

            elif "```FULL_TEXT" in resp:
                query = extract_prompt(resp, "FULL_TEXT")
                yield f"阅读论文编号:\n{query}"
                full_text = arx_eng.retrieve_full_paper_text(query)
                feedback = f"```EXPIRATION {self.arxiv_paper_exp_time}\n{full_text}```"
                yield f"论文全文:\n{feedback}"

            elif "```ADD_PAPER" in resp:
                query = extract_prompt(resp, "ADD_PAPER")
                feedback, text = self.phd.add_review(query, arx_eng)
                if len(self.reference_papers) < self.num_ref_papers:
                    self.reference_papers.append(text)
                yield f"知识入库:\n{query}"

            if len(self.phd.lit_review) >= self.num_papers_lit_review:
                lit_review_sum = self.phd.format_review()
                yield f"知识预览:\n{lit_review_sum}"
                self.phd.lit_review = []
                return False

            resp = self.phd.inference(
                self.research_topic, "literature review",
                feedback=feedback, step=_i + 1, temp=0.4
            )


if __name__ == '__main__':
    lab = LaboratoryWorkflow(research_topic="数字化经济", openai_api_key="")
    for chunk in lab.literature_review():
        print(chunk)