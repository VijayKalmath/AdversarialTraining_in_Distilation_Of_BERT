# Quiet TensorFlow.
import os
import sys
sys.path.append('../utils')
from ganbert_utils import *
from ganbert_models import *

import numpy as np
from transformers import AutoTokenizer # TFAutoModelForSequenceClassification, pipeline

from textattack import Attacker,AttackArgs
from tokenizers import Tokenizer
from textattack.datasets import HuggingFaceDataset
from textattack.models.wrappers import ModelWrapper

class InferenceGANBert(nn.Module):
    
    def __init__(self, transformer, discriminator):
        super().__init__()
        self.device = get_gpu_details()
        self.transformer = transformer
        self.transformer.eval() 
        self.discriminator = discriminator
        self.discriminator.eval()
        
        if torch.cuda.is_available(): 
            self.discriminator.cuda()
            self.transformer.cuda()
    
            if True:
                self.transformer = torch.nn.DataParallel(self.transformer)
        
    def forward(self, inp_id_mask):
        
        
#         move to device GPU
        inp_id_mask[0].to(self.device)
        inp_id_mask[1].to(self.device)
                
        # do the forward pass
        with torch.no_grad():        
            model_outputs = self.transformer(inp_id_mask[0], attention_mask=inp_id_mask[1])
            hidden_states = model_outputs[-1]
            _, _, probs = self.discriminator(hidden_states)
            
        return probs
            


class BertModelWrapper(ModelWrapper):
    """Transformers sentiment analysis pipeline returns a list of responses,
    like
        [{'label': 'POSITIVE', 'score': 0.7817379832267761}]
    We need to convert that to a format TextAttack understands, like
        [[0.218262017, 0.7817379832267761]
    """

    def __init__(self, model_path):
        
        model_dict = torch.load(model_path)
        model_name = 'bert-base-cased'
        self.transformer = AutoModel.from_pretrained(model_name)
#         self.transformer.load_state_dict(model_dict['bert_encoder'])
        self.transformer.eval()
        print("type of transformer: ", type(self.transformer))
        
        hidden_size_bert = AutoConfig.from_pretrained(model_name).hidden_size
        print("hiddensize is: ", hidden_size_bert)
        hidden_size_bert = int(hidden_size_bert)
        
        self.discriminator = Discriminator(input_size=hidden_size_bert, hidden_sizes=[hidden_size_bert], num_labels=3, dropout_rate=0.1)
        #self.discriminator = nn.Module()
        self.discriminator.load_state_dict(model_dict['discriminator'])
        self.discriminator.eval()
        print("type of discriminator: ", type(self.discriminator))
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        self.model = InferenceGANBert(self.transformer, self.discriminator)

    def __call__(self, test_inputs):
        
        # convert the text_inputs by passing into dataloader
        
        test_dataloader = [[],[]]
        
        for sent in test_inputs : 
        
            # encode the text
            test_dataloader[0].append((self.tokenizer.encode(sent, add_special_tokens=True, max_length=64, padding="max_length", truncation=True)))
        
            # attention
            test_dataloader[1].append(([int(token_id > 0) for token_id in test_dataloader[0][-1]]))
        
        # unwrap the model contents and do the actual computation
        return self.model(torch.tensor(test_dataloader))