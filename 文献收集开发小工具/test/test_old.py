import os
import re
import time
import arxiv
from pypdf import PdfReader
from openai import OpenAI

class ArxivSearch:
    def __init__(self):
        # Construct the default API client.
        self.sch_engine = arxiv.Client()
        self.max_query_len = 300
        
    def _process_query(self, query: str) -> str:
        """Process query string to fit within MAX_QUERY_LENGTH while preserving as much information as possible"""
        MAX_QUERY_LENGTH = 300
        
        if len(query) <= MAX_QUERY_LENGTH:
            return query
        
        # Split into words
        words = query.split()
        processed_query = []
        current_length = 0
        
        # Add words while staying under the limit
        # Account for spaces between words
        for word in words:
            # +1 for the space that will be added between words
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
                    sort_by=arxiv.SortCriterion.Relevance)

                paper_sums = list()
                # `results` is a generator; you can iterate over its elements one by one...
                for r in self.sch_engine.results(search):
                    paperid = r.pdf_url.split("/")[-1]
                    pubdate = str(r.published).split(" ")[0]
                    paper_sum = f"Title: {r.title}\n"
                    paper_sum += f"Summary: {r.summary}\n"
                    paper_sum += f"Publication Date: {pubdate}\n"
                    #paper_sum += f"Categories: {' '.join(r.categories)}\n"
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
        # Download the PDF to the PWD with a custom filename.
        paper.download_pdf(filename="downloaded-paper.pdf")
        # creating a pdf reader object
        reader = PdfReader('downloaded-paper.pdf')
        # Iterate over all the pages
        for page_number, page in enumerate(reader.pages, start=1):
            # Extract text from the page
            try:
                text = page.extract_text()
            except Exception as e:
                os.remove("downloaded-paper.pdf")
                time.sleep(2.0)
                return "EXTRACTION FAILED"

            # Do something with the text (e.g., print it)
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
        text = text.replace("```\n", "```")
        return text

    def override_inference(self, query, temp=0.0):
        sys_prompt = f"""You are {self.role_description()}"""
        model_resp = query_model(model_str=self.model, system_prompt=sys_prompt, prompt=query, temp=temp, openai_api_key=self.openai_api_key)
        return model_resp

    def inference(self, research_topic, phase, step, feedback="", temp=None):
        sys_prompt = f"""You are {self.role_description()} \nTask instructions: {self.phase_prompt()}\n{self.command_descriptions(phase)}"""
        history_str = "\n".join([_[1] for _ in self.history])
        complete_str = str()
        if step/(self.max_steps-1) > 0.7: complete_str = "You must finish this task and submit as soon as possible!"
        prompt = (
            f"""{'~' * 10}\nHistory: {history_str}\n{'~' * 10}\n"""
            f"Current Step #{step}, Phase: {phase}\n{complete_str}\n"
            f"[Objective] Your goal is to perform research on the following topic: {research_topic}\n"
            f"Feedback: {feedback}\nYour previous command was: {self.prev_comm}. Make sure your new output is very different.\nPlease produce a single command below:\n")
        model_resp = query_model(model_str=self.model, system_prompt=sys_prompt, prompt=prompt, temp=temp, openai_api_key=self.openai_api_key)
        model_resp = self.clean_text(model_resp)
        self.prev_comm = model_resp
        steps_exp = None
        if feedback is not None and "```EXPIRATION" in feedback:
            steps_exp = int(feedback.split("\n")[0].replace("```EXPIRATION ", ""))
            feedback = extract_prompt(feedback, "EXPIRATION")
        self.history.append((steps_exp, f"Step #{step}, Phase: {phase}, Feedback: {feedback}, Your response: {model_resp}"))
        # remove histories that have expiration dates
        for _i in reversed(range(len(self.history))):
            if self.history[_i][0] is not None:
                self.history[_i] = (self.history[_i][0] - 1, self.history[_i][1])
                if self.history[_i][0] < 0:
                    self.history.pop(_i)
        if len(self.history) >= self.max_hist_len:
            self.history.pop(0)
        return model_resp

    def reset(self):
        self.history.clear()  # Clear the deque
        self.prev_comm = ""

    
class PhDStudentAgent(BaseAgent):
    def __init__(self):
        super().__init__()  
        self.phases = [
            "literature review"
        ]
        self.lit_review = []


    def requirements_txt(self):
        sys_prompt = f"""You are {self.role_description()} \nTask instructions: Your goal is to integrate all of the knowledge, code, reports, and notes provided to you and generate a requirements.txt for a github repository for all of the code."""
        history_str = "\n".join([_[1] for _ in self.history])
        prompt = (
            f"""History: {history_str}\n{'~' * 10}\n"""
            f"Please produce the requirements.txt below in markdown:\n")
        model_resp = query_model(model_str=self.model, system_prompt=sys_prompt, prompt=prompt, openai_api_key=self.openai_api_key)
        return model_resp

    def example_command(self, phase):
        if phase not in self.phases:
            raise Exception(f"Invalid phase: {phase}")
        return ()

    def command_descriptions(self):
        return (
                "To collect paper summaries, use the following command: ```SUMMARY\nSEARCH QUERY\n```\n where SEARCH QUERY is a string that will be used to find papers with semantically similar content and SUMMARY is just the word SUMMARY. Make sure your search queries are very short.\n"
                "To get the full paper text for an arXiv paper, use the following command: ```FULL_TEXT\narXiv paper ID\n```\n where arXiv paper ID is the ID of the arXiv paper (which can be found by using the SUMMARY command), and FULL_TEXT is just the word FULL_TEXT. Make sure to read the full text using the FULL_TEXT command before adding it to your list of relevant papers.\n"
                "If you believe a paper is relevant to the research project proposal, you can add it to the official review after reading using the following command: ```ADD_PAPER\narXiv_paper_ID\nPAPER_SUMMARY\n```\nwhere arXiv_paper_ID is the ID of the arXiv paper, PAPER_SUMMARY is a brief summary of the paper, and ADD_PAPER is just the word ADD_PAPER. You can only add one paper at a time. \n"
                "Make sure to use ADD_PAPER when you see a relevant paper. DO NOT use SUMMARY too many times."
                "You can only use a single command per inference turn. Do not use more than one command per inference. If you use multiple commands, then only one of them will be executed, not both.\n"
                "Make sure to extensively discuss the experimental results in your summary.\n"
                "When performing a command, make sure to include the three ticks (```) at the top and bottom ```COMMAND\ntext\n``` where COMMAND is the specific command you want to run (e.g. ADD_PAPER, FULL_TEXT, SUMMARY). Do not use the word COMMAND make sure to use the actual command, e.g. your command should look exactly like this: ```ADD_PAPER\ntext\n``` (where the command could be from ADD_PAPER, FULL_TEXT, SUMMARY)\n"
                )
        
    def phase_prompt(self):
        phase_str = (
                "Your goal is to perform a literature review for the presented task and add papers to the literature review.\n"
                "You have access to arXiv and can perform two search operations: (1) finding many different paper summaries from a search query and (2) getting a single full paper text for an arXiv paper.\n"
            )
        rev_papers = "Papers in your review so far: " + " ".join([_paper["arxiv_id"] for _paper in self.lit_review])
        phase_str += rev_papers if len(self.lit_review) > 0 else ""
        return phase_str


    def role_description(self):
        return "a computer science PhD student at a top university."

    def add_review(self, review, arx_eng, agentrxiv=False, GLOBAL_AGENTRXIV=None):
        try:
            if agentrxiv:
                arxiv_id = review.split("\n")[0]
                review_text = "\n".join(review.split("\n")[1:])
                full_text = GLOBAL_AGENTRXIV.retrieve_full_text(arxiv_id,)
            else:
                arxiv_id, review_text = review.strip().split("\n", 1)
                full_text = arx_eng.retrieve_full_paper_text(arxiv_id)
            review_entry = {
                "arxiv_id": arxiv_id,
                "full_text": full_text,
                "summary": review_text,
            }
            self.lit_review.append(review_entry)
            return f"Successfully added paper {arxiv_id}", full_text
        except Exception as e:
            return f"Error trying to add review -- bad formatting, try again: {str(e)}. Your provided Arxiv ID might not be valid. Make sure it references a real paper, which can be found using the SUMMARY command.", ""

    def format_review(self):
        return "Provided here is a literature review on this topic:\n" + "\n".join(
            f"arXiv ID: {_l['arxiv_id']}, Summary: {_l['summary']}"
            for _l in self.lit_review)
    
       
def extract_prompt(text, word):
    code_block_pattern = rf"```{word}(.*?)```"
    code_blocks = re.findall(code_block_pattern, text, re.DOTALL)
    extracted_code = "\n".join(code_blocks).strip()
    return extracted_code



def query_model(model_str, prompt, system_prompt, tries=5,  temp=None):
    for _ in range(tries):
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}]
     
                deepseek_client = OpenAI(
                        api_key='sk-cfaf9101c8244f4bb274cb5ea4dbcf26',
                        base_url="https://api.deepseek.com/v1"
                    )
                if temp is None:
                        completion = deepseek_client.chat.completions.create(
                            model=model_str,
                            messages=messages)
                else:
                        completion = deepseek_client.chat.completions.create(
                            model=model_str,
                            messages=messages,
                            temperature=temp)
                answer = completion.choices[0].message.content
                return answer
    
class LaboratoryWorkflow:
    def __init__(self, research_topic='', openai_api_key='',
                 max_steps=100, num_papers_lit_review=3 
             ):
        self.max_steps = max_steps
        self.openai_api_key = openai_api_key
        self.research_topic = research_topic
        self.num_papers_lit_review = num_papers_lit_review
        self.phd = PhDStudentAgent()



    def literature_review(self):
            """
            Perform literature review phase
            @return: (bool) whether to repeat the phase
            """
            arx_eng = ArxivSearch()
            max_tries = self.max_steps # lit review often requires extra steps
            # get initial response from PhD agent
            resp = self.phd.inference(self.research_topic, "literature review", step=0, temp=0.4)
    
            # iterate until max num tries to complete task is exhausted
            for _i in range(max_tries):
               
                feedback = str()
                # grab summary of papers from arxiv
                if "```SUMMARY" in resp:
                    query = extract_prompt(resp, "SUMMARY")
                    yield  f"关键词:\n{query}"
                    papers = arx_eng.find_papers_by_str(query, N=self.arxiv_num_summaries)
                    yield   f"摘要检索:\n{papers}"
                    feedback = f"You requested arXiv papers related to the query {query}, here was the response\n{papers}"
                    
                # grab full text from arxiv ID
                elif "```FULL_TEXT" in resp:
                    query = extract_prompt(resp, "FULL_TEXT")
                    yield  f"阅读论文编号:\n{query}"
                    full_text = arx_eng.retrieve_full_paper_text(query)
                    # expiration timer so that paper does not remain in context too long
                    arxiv_paper = f"```EXPIRATION {self.arxiv_paper_exp_time}\n" + full_text + "```"
                    feedback = arxiv_paper
                    yield  f"论文全文:\n{arxiv_paper}"
                # if add paper, extract and add to lit review, provide feedback
                elif "```ADD_PAPER" in resp:
                    query = extract_prompt(resp, "ADD_PAPER")      
                    feedback, text = self.phd.add_review(query, arx_eng)
                    if len(self.reference_papers) < self.num_ref_papers:
                        self.reference_papers.append(text)
                    yield  f"知识入库:\n{query}"
                    
                # completion condition
                if len(self.phd.lit_review) >= self.num_papers_lit_review:
                    # generate formal review
                    lit_review_sum = self.phd.format_review()
                    # if human in loop -> check if human is happy with the produced review
                    yield  f"知识预览:\n{lit_review_sum}"
                    self.phd.lit_review = []
                    return False
                resp = self.phd.inference(self.research_topic, "literature review", feedback=feedback, step=_i + 1, temp=0.4)
              

            
if __name__ == '__main__':
    lab = LaboratoryWorkflow(
        research_topic = "数字化经济",
        openai_api_key = ""
        )
    for chunk in lab.literature_review():
         print(chunk)
    














