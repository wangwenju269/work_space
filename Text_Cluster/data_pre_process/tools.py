from transformers import BertTokenizer
import sys
sys.path.append('./')
from utils.arguments_parse import args
def get_tokenizer():
    """添加特殊中文字符和未使用的token【unused1】"""
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_path)
    special_tokens_dict = {'additional_special_tokens':['”','“']}
    tokenizer.add_special_tokens(special_tokens_dict)
    return tokenizer
tokenizer= get_tokenizer()


if __name__ == '__main__':
   print(tokenizer)