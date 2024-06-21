```shell


deepspeed --include="localhost:2,3,6" --master_port 2411  /data/wangwj-t/workspace/LLaMA-Factory/src/train.py \
    --deepspeed   /data/wangwj-t/workspace/LLaMA-Factory/examples/deepspeed/ds_z3_config.json \
    --stage          pt \
    --do_train  \
    --do_eval  \
    --val_size               0.2  \
    --eval_steps             10  \
    --evaluation_strategy    steps  \
    --dataset               gongwen_writer  \
    --model_name_or_path    /data/public/LLM/basemodels/qwen/Qwen-14B-Chat  \
    --finetuning_type   freeze  \
    --num_layer_trainable  1   \
    --template          qwen  \
    --per_device_train_batch_size  2  \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps  2  \
    --lr_scheduler_type cosine  \
    --logging_steps    10  \
    --learning_rate    1e-5   \
    --plot_loss  \
    --gradient_checkpointing  \
    --save_strategy          epoch  \
    --num_train_epochs       4  \
    --cutoff_len 8192    \
    --bf16  \
    --output_dir    /data/wangwj-t/workspace/WriteReport/checkpoint/pt2 \
    --dataset_dir   /data/wangwj-t/workspace/LLaMA-Factory/data  
  
  
  
  
  
  
 CUDA_VISIBLE_DEVICES=1  python src/api_demo.py     --model_name_or_path /data/wangwj-t/workspace/WriteReport/checkpoint/export  --template qwen    
  
  
  
  curl -X POST http://0.0.0.0:8000/v1/chat/completions -H "content-type:application/json" -d '{
  "messages":[{"role":"user","content":"xinqingbuhao,请将拼音转化汉字"}],
  "model": "qwen",
  "stream": false,
  "max_tokens": 256
}'
 
 


   
  
  
deepspeed --include="localhost:1,2,3,5"   /data/wangwj-t/workspace/LLaMA-Factory/src/train.py \
    --deepspeed   /data/wangwj-t/workspace/LLaMA-Factory/examples/deepspeed/ds_z2_config.json \
    --stage          sft \
    --do_train  \
    --do_eval  \
    --val_size               0.2  \
    --eval_steps             10  \
    --evaluation_strategy    steps  \
    --dataset               id \
    --model_name_or_path       /data/wangwj-t/workspace/WriteReport/checkpoint/pt/checkpoint-12   \
    --finetuning_type        lora        \
    --lora_rank 8  \
    --lora_target       c_proj,c_attn    \
    --template          qwen    \
    --per_device_train_batch_size  4  \
    --per_device_eval_batch_size 2   \
    --gradient_accumulation_steps  4  \
    --lr_scheduler_type cosine  \
    --logging_steps    10  \
    --learning_rate    1e-5  \
    --plot_loss  \
    --gradient_checkpointing  \
    --save_strategy          epoch  \
    --num_train_epochs       8  \
    --cutoff_len  2048     \
    --overwrite_cache       \
    --overwrite_output_dir  \
    --output_dir    /data/wangwj-t/workspace/WriteReport/checkpoint/sft_last     \
    --dataset_dir   /data/wangwj-t/workspace/LLaMA-Factory/data  
    
    

    
    
    
    
    
    CUDA_VISIBLE_DEVICES=2  python src/export_model.py  \
    --model_name_or_path    /data/wangwj-t/workspace/WriteReport/checkpoint/pt/checkpoint-12  \
    --adapter_name_or_path  /data/wangwj-t/workspace/WriteReport/checkpoint/sft_last/checkpoint-51  \
    --template qwen  \
    --finetuning_type lora \
    --export_dir /data/wangwj-t/workspace/WriteReport/checkpoint/export  \
    --export_size 2 \
    --export_legacy_format False
    
    

             
                
```



