import torch.nn as nn
from transformers import (OPENAI_GPT_PRETRAINED_CONFIG_ARCHIVE_MAP, BERT_PRETRAINED_CONFIG_ARCHIVE_MAP,
                          XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP, ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP)
try:
    from transformers import ALBERT_PRETRAINED_CONFIG_ARCHIVE_MAP
except:
    pass
from transformers import AutoModel, BertModel, BertConfig
from utils.layers import *

MODEL_CLASS_TO_NAME = {
    'gpt': list(OPENAI_GPT_PRETRAINED_CONFIG_ARCHIVE_MAP.keys()),
    'bert': list(BERT_PRETRAINED_CONFIG_ARCHIVE_MAP.keys()),
    'xlnet': list(XLNET_PRETRAINED_CONFIG_ARCHIVE_MAP.keys()),
    'roberta': list(ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP.keys()),
}
try:
    MODEL_CLASS_TO_NAME['albert'] =  list(ALBERT_PRETRAINED_CONFIG_ARCHIVE_MAP.keys())
except:
    pass

MODEL_NAME_TO_CLASS = {model_name: model_class for model_class, model_name_list in MODEL_CLASS_TO_NAME.items() for model_name in model_name_list}

#Add SapBERT configuration
model_name = "./LM_models/cambridgeltl--SapBERT-from-PubMedBERT-fulltext" # 'cambridgeltl/SapBERT-from-PubMedBERT-fulltext'
MODEL_NAME_TO_CLASS[model_name] = 'bert'
model_name = 'cambridgeltl/SapBERT-from-PubMedBERT-fulltext'
MODEL_NAME_TO_CLASS[model_name] = 'bert'

class LM(nn.Module):
    valid_model_types = set(MODEL_CLASS_TO_NAME.keys())

    def __init__(self, model_name, output_token_states=False, from_checkpoint=None):
        super().__init__()
        self.model_type = MODEL_NAME_TO_CLASS[model_name]
        self.output_token_states = output_token_states
        assert not self.output_token_states or self.model_type in ('bert', 'roberta', 'albert')

        model_class = AutoModel
        self.module = model_class.from_pretrained(model_name, output_hidden_states=True)
        if from_checkpoint is not None:
            self.module = self.module.from_pretrained(from_checkpoint, output_hidden_states=True)
        self.sent_dim = self.module.config.hidden_size

    def forward(self, *inputs, layer_id=-1):
        '''
        output_token_states: if True, return hidden states of specific layer and attention masks
        '''
        input_ids, attention_mask, token_type_ids, output_mask = inputs
        outputs = self.module(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        all_hidden_states = outputs[-1]
        hidden_states = all_hidden_states[layer_id]

        if self.model_type in ('albert',):
            if self.output_token_states:
                return hidden_states, output_mask
            context_emb = hidden_states[:, 0]
        else:  # bert / roberta
            if self.output_token_states:
                return hidden_states, output_mask
            context_emb = self.module.pooler(hidden_states)
        return context_emb, all_hidden_states