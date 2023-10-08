import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel
from data_pre_process.tools import tokenizer

class Bert_classify(BertPreTrainedModel):
    def __init__(self, config, params):
        super().__init__(config)
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_class = params.num_class
        self.bert = BertModel(config= self.config)
        self.bert.resize_token_embeddings(len(tokenizer))
        self.classifier = nn.Linear(self.hidden_size, self.num_class)
        #[768,9]
    def forward(self, input_ids,attention_mask,token_type_ids):
        output = self.bert(input_ids=input_ids,attention_mask=attention_mask,
                           token_type_ids=token_type_ids)
        cls_hidden_state = output.pooler_output
        # [bs, hidden]
        # size:[16,  768] * [768,9] = [16, 9]
        # [16,1]
        # [16,9]

        logits = self.classifier(cls_hidden_state)
        # [bs, num]  [16,9]

        return logits

