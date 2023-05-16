import torch
from torch import nn
from transformers import AutoConfig
from transformers.models.marian import MarianPreTrainedModel, MarianModel

class MarianForMT(MarianPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = MarianModel(config)
        target_vocab_size = config.decoder_vocab_size
        self.register_buffer("final_logits_bias", torch.zeros((1, target_vocab_size)))
        self.lm_head = nn.Linear(config.d_model, target_vocab_size, bias=False)
        self.post_init()

    def forward(self,input_ids ,attention_mask,decoder_input_ids,labels):
        output = self.model(input_ids ,attention_mask,decoder_input_ids,labels)
        sequence_output = output.last_hidden_state
        lm_logits = self.lm_head(sequence_output) + self.final_logits_bias
        return lm_logits
