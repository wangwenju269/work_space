from torch import nn
from transformers import BertPreTrainedModel,BertModel
class  BertForPairwiseCLS(BertPreTrainedModel):
       def __init__(self,config,args):
           super().__init__(config)
           self.bert = BertModel(config)
           self.dropout = nn.Dropout(config.hidden_dropout_prob)
           self.classifier = nn.Linear(config.hidden_size, args.num_labels)
           self.post_init()

       def forward(self, input_ids, attention_mask, token_type_ids):
           bert_output = self.bert( input_ids=input_ids, attention_mask=attention_mask,
                                    token_type_ids=token_type_ids)

           cls_vectors = bert_output.last_hidden_state[:, 0, :]
           cls_vectors = self.dropout(cls_vectors)
           logits = self.classifier(cls_vectors)
           return logits
