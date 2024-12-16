import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


class ModelLLM:
    def __init__(self, cfg):
        self.model_cls = cfg.get('model_cls', AutoModelForCausalLM)
        self.tokenizer_cls = cfg.get('tokenizer_cls', AutoTokenizer)

        self.device_map = cfg.get('device_map', 'auto')
        self.generation_cfg = GenerationConfig(
            **cfg.get('generate_cfg', {}))

        self.custom_chat = cfg.get('custom_chat', False)

        self.end_token = cfg.get('end_token', '<|endofthink|>')
        self.include_end = cfg.get('include_end', True)

        self.setup()
        # self.agent_type = self.cfg.get('agent_type', AgentType.REACT)

    def setup(self):
        model_cls = self.model_cls
        tokenizer_cls = self.tokenizer_cls
        self.model = model_cls.from_pretrained(
            self.model_dir,
            device_map=self.device_map,
            # device='cuda:0',
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        self.tokenizer = tokenizer_cls.from_pretrained(
            self.model_dir, trust_remote_code=True
        )
        self.model = self.model.eval()

    def generate(self, prompt, functions: list = [], stop_words=None, **kwargs):
        if stop_words is None:
            stop_words = ['Observation:', 'Observation:\n']

        stop_words_ids = [self.tokenizer.encode(w) for w in stop_words]
        if self.custom_chat and self.model.chat:
            response = self.model.chat(
                self.tokenizer,
                prompt,
                stop_words_ids=stop_words_ids,
                history=None,
                system='You are a helpful assistant.'
            )[0]
        else:
            response = self.chat(prompt)

        end_idx = response.find(self.end_token)
        if end_idx != -1:
            end_idx += len(self.end_token) if self.include_end else 0
            response = response[:end_idx]

        return response

    def chat(self, prompt):
        device = self.model.device
        input_ids = self.tokenizer(
            prompt, return_tensors='pt').input_ids.to(device)
        input_len = input_ids.shape[1]

        result = self.model.generate(
            input_ids=input_ids, generation_config=self.generation_cfg
        )

        result = result[0].tolist()[input_len:]
        response = self.tokenizer.decode(result)

        return response


class ModelScopeChatGLM(ModelLLM):
    def chat(self, prompt):
        device = self.model.device
        input_ids = self.tokenizer(
            prompt, return_tensors='pt').input_ids.to(device)
        input_len = input_ids.shape[1]

        eos_token_id = [
            self.tokenizer.eos_token_id,
            self.tokenizer.get_command('<|user|>'),
            self.tokenizer.get_command('<|observation|>')
        ]
        result = self.model.generate(
            input_ids=input_ids,
            generation_config=self.generation_cfg,
            eos_token_id=eos_token_id
        )

        result = result[0].tolist()[input_len:]
        response = self.tokenizer.decode(result)
        # 遇到生成'<', '|', 'user', '|', '>'的case
        response = response.split('<|user|>')[0].split('<|observation|>')[0]
        return response