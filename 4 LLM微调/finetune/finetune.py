from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import transformers
from dataset import DataCollatorForLMDataset, SupervisedDataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer
from trainer import UserTrainer


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="THUDM/glm-4-9b")


@dataclass
class DataArguments:
    train_file: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    validation_file: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )


@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    max_length: int = field(
        default=8192,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    truncate_side: str = field(default="left", metadata={"help": "left or right"})
    llm_type: str = field(default="glm4")
    use_lora: Optional[bool] = field(default=True)


@dataclass
class LoraArguments:
    lora_r: int = 64
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: str = 'all-linear'
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False
    lora_modules_to_save: str = ""
    lora_layer_replication: Optional[List[Tuple[int, int]]] = None
    lora_layers_to_transform: Optional[List[int]] = None
    lora_layers_pattern: Optional[str] = None


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """
    Create a supervised data module for training and evaluation.

    Args:
        tokenizer (`transformers.PreTrainedTokenizer`):
            The tokenizer to preprocess the data.
        data_args:
            Arguments related to data loading and processing.

    Returns:
        `Dict`: A dictionary containing the training and evaluation datasets.
    """
    # Load and process the training dataset
    train_dataset = open(data_args.train_file, encoding="utf-8").readlines()
    train_dataset = SupervisedDataset(tokenizer=tokenizer, raw_data=train_dataset)

    # Load and process the evaluation dataset if provided
    if data_args.validation_file:
        eval_dataset = open(data_args.validation_file, encoding="utf-8").readlines()
        eval_dataset = SupervisedDataset(tokenizer=tokenizer, raw_data=eval_dataset)
    else:
        eval_dataset = None

    # Return the datasets as a dictionary
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


def get_parameter_number(model):
    trainable_params, all_param = 0, 0
    for param in model.parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return {"Total": all_param, "Trainable": trainable_params, "ratio": trainable_params / all_param}


def train():
    # Parse arguments
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    model_args, data_args, training_args, lora_args = parser.parse_args_into_dataclasses()

    # Determine compute data type
    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )
    print(model_args.model_name_or_path)
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=compute_dtype,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=True
    )

    if training_args.use_lora:
        for name, param in model.named_parameters():
            param.requires_grad = False
            
        if  training_args.gradient_checkpointing:
            model.enable_input_require_grads()
            
        if  lora_args.q_lora:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )

        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_args.lora_target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            layers_to_transform=lora_args.lora_layers_to_transform,
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, lora_config)
        # model.base_model.model.transformer.embedding.word_embeddings.requires_grad_(True)  # embeding开启梯度

    print(get_parameter_number(model))

    # Load data
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    # Define data collator
    collate_fn = DataCollatorForLMDataset(max_length=training_args.max_length)

    # Initialize trainer
    trainer = UserTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=collate_fn,
        **data_module
    )

    # Train the model
    trainer.train()

    # Save the model
    trainer.save_model()

if __name__ == "__main__":
    train()