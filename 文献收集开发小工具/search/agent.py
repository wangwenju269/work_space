import os
import re
from typing import List, Dict, Generator, Optional, Tuple
from openai import OpenAI
from dotenv import load_dotenv

# 加载.env文件
load_dotenv()

class BaseAgent:
    """基础智能体类"""
    
    def __init__(
        self,
        model: str = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None, 
        max_steps: int = 100,
        max_hist_len: int = 6
    ):
        self.model = model or os.environ.get("MODEL")
        self.api_key = api_key or os.environ.get("API_KEY")
        self.base_url = base_url or os.environ.get("BASE_URL")
        
        self.max_steps = max_steps
        self.max_hist_len = max_hist_len
        self.history: List[Tuple[Optional[int], str]] = []
        self.previous_command = ""
        

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

    def phase_prompt(self) -> str:
        """阶段提示模板"""
        return (
            "You are working on a research project. Current phase is {phase}.\n"
            "Follow these rules:\n"
            "1. Use precise commands to interact with the system\n"
            "2. Analyze previous feedback carefully\n"
            "3. Validate data before proceeding to next step"
        )

    def command_descriptions(self) -> str:
        """命令描述"""
        return (
            "Available Commands:\n"
            "/search <query> - Search for information\n"
            "/analyze <data> - Perform data analysis\n"
            "/submit <results> - Final submission\n"
            "Format commands in code blocks when needed"
        )

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
            f"Task instructions: {self.phase_prompt().format(phase=phase)}\n"  # 格式化阶段提示
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
        self.history.append((None, f"Step #{step}, Phase: {phase}, Your response: {response}"))
        # 保持历史记录长度限制
        self._trim_history()
        return self.clean_response(response)

    def _trim_history(self):
        """保持历史记录长度限制"""
        while len(self.history) > self.max_hist_len:
            self.history.pop(0)
        
        
    def extract_prompt(self, text: str, word: str) -> str:
        code_block_pattern = rf"```{word}(.*?)```"
        code_blocks = re.findall(code_block_pattern, text, re.DOTALL)
        extracted_code = "\n".join(code_blocks).strip()
        return extracted_code  

    def _query_model(
        self,
        system_prompt: str,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: int = 2000
    ) -> str:
        """调用模型API"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

        params = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens
        }
        if temperature is not None:
            params["temperature"] = temperature

        try:
            response = client.chat.completions.create(**params)
            return response.choices[0].message.content
        except Exception as e:
            return f"API Error: {str(e)}"