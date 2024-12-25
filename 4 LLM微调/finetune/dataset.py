from dataclasses import dataclass
from typing import Sequence, Dict
import json

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


@dataclass
class DataCollatorForLMDataset:
    padding_value: int = -100
    max_length: int = 8192
    truncate_side: str = 'left'

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        padding_value = self.padding_value
        max_length = self.max_length
        truncate_side = self.truncate_side
        batch_input_ids, batch_labels = [], []

        for instance in instances:
            input_ids = instance["input_ids"]
            labels = instance["labels"]

            # Truncate sequences if they exceed max_length
            if input_ids.shape[0] > max_length:
                if truncate_side == 'right':
                    input_ids = input_ids[:max_length]
                    labels = labels[:max_length]
                elif truncate_side == 'left':
                    cut_num = input_ids.shape[0] - max_length
                    input_ids = torch.cat([input_ids[:2], input_ids[2 + cut_num:]], dim=0)
                    labels = torch.cat([labels[:2], labels[2 + cut_num:]], dim=0)

            batch_input_ids.append(input_ids)
            batch_labels.append(labels)
            
        del input_ids
        del labels   
        # Pad sequences to the same length
        input_ids = pad_sequence(batch_input_ids, batch_first=True)
        labels = pad_sequence(batch_labels, batch_first=True, padding_value=padding_value)

        # Find the EOS indices and determine the maximum position
        eos_indices = input_ids.argmin(dim=1) - 1
        max_position = eos_indices.max()

        # Truncate sequences based on the maximum position
        if max_position > 0:
            input_ids = input_ids[:, : max_position + 1]
            labels = labels[:, : max_position + 1]

        return dict(
            input_ids=input_ids,
            labels=labels,
            use_cache = False
        )

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        raw_data: str
    ):
        super().__init__()
        self.raw_data = raw_data
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.raw_data)

    def __getitem__(self, idx) -> Dict[str, list]:
        conversation = json.loads(self.raw_data[idx])["messages"]
        inputs, labels = conversation_to_ids_glm4(
            conversation=conversation,
            tokenizer=self.tokenizer
        )
        return dict(
            input_ids=inputs,
            labels=labels
        )
        
        
def conversation_to_ids(conversation, tokenizer):
    # Apply chat template and tokenize the conversation
    input_ids = tokenizer.apply_chat_template(conversation, tokenize=True)
    input_ids = torch.tensor(input_ids + [tokenizer.eos_token_id])
    
    # Initialize labels tensor with -100
    labels = torch.full_like(input_ids, -100)
    
    # Find indices where the assistant and user tokens start
    starts = torch.where(
        input_ids == tokenizer.convert_tokens_to_ids("<|assistant|>")
    )[0]
    
    ends = torch.where(
        input_ids == tokenizer.convert_tokens_to_ids("<|user|>")
    )[0]
    
    # Adjust ends to include the last segment
    ends = torch.cat((ends[1:], torch.tensor([len(input_ids)])))
    
    # Assign labels for the assistant's responses
    for start, end in zip(starts, ends):
        labels[start + 2:end] = input_ids[start + 2:end]  # Skip "<|assistant|>", "\n"
    
    return input_ids, labels

def conversation_to_ids_glm4(conversation, tokenizer):
    input_ids = []
    starts = []
    ends = []

    for item in conversation:
        role = item["role"]
        content = item["content"]
        assert role in ["user", "assistant"]

        if role == "assistant" and content != '':
            starts.append(len(input_ids))

        input_ids.extend(tokenizer.build_single_message(role, item.get("metadata", ""), content))

        if role == 'assistant' and content != '':
            ends.append(len(input_ids))

    # Append EOS token
    input_ids.append(tokenizer.eos_token_id)
    # Append prefix token
    input_ids = torch.tensor(tokenizer.get_prefix_tokens() + input_ids)
    labels = torch.full_like(input_ids, -100)

    for start, end in zip(starts, ends):
        labels[start + 3:end + 3] = input_ids[start + 3:end + 3]

    return input_ids, labels