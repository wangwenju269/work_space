#!/bin/bash

# Define variables
MODEL="ZhipuAI/glm-4-9b-chat"
DATA="data/train.jsonl"
EVAL_DATA="/data/dev.jsonl"
LLM_TYPE="glm4"
OUTPUT_DIR="data/output"

# Disable NCCL P2P and IB for compatibility
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# Run training with DeepSpeed
deepspeed --include="localhost:0" --master_port 2411 finetune.py \
    --model_name_or_path "$MODEL" \
    --llm_type "$LLM_TYPE" \
    --train_file "$DATA" \
    --validation_file "$EVAL_DATA" \
    --output_dir "$OUTPUT_DIR" \
    --remove_unused_columns false \
    --prediction_loss_only false \
    --bf16 true \
    --bf16_full_eval true \
    --fp16 false \
    --fp16_full_eval false \
    --do_train \
    --do_eval \
    --max_length 8192 \
    --logging_dir ./logs/ \
    --logging_strategy "steps" \
    --logging_steps 1 \
    --num_train_epochs 4 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "steps" \
    --save_strategy "steps" \
    --eval_steps 100 \
    --save_steps 400 \
    --save_total_limit 10 \
    --learning_rate 1e-5 \
    --weight_decay 0.1 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --gradient_checkpointing true \
    --deepspeed stage3.json \
    --run_name glm4_longwriter \
    --group_by_length true \
    --use_lora true \
    --lora_target_modules "transformer.encoder.layers\.\d+\.self_attention.query_key_value"