from peft import PeftModel
import torch
from transformers import AutoModel, AutoTokenizer

# Define paths
model_dir = "ZhipuAI/glm-4-9b-chat"
path_to_adapter = "data/output/checkpoint-400"
merge_path = "data/merge"

# Load the base model
model = AutoModel.from_pretrained(
    model_dir,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="cuda:0"
)

# Load the adapter and merge it with the base model
merge_model = PeftModel.from_pretrained(
    model,
    path_to_adapter,
    device_map="auto",
    trust_remote_code=True
).eval()

merge_model = merge_model.merge_and_unload()

# Save the merged model
merge_model.save_pretrained(merge_path, safe_serialization=False)

# Load and save the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
tokenizer.save_pretrained(merge_path)