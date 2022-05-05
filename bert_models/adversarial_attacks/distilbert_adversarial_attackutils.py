import os
import sys
sys.path.append('../utils')

from distilbert_utils import *
from distilbert_finetuning import *

import numpy as np
from transformers import AutoTokenizer, DistilBertTokenizer, DistilBertForSequenceClassification 

from textattack import Attacker,AttackArgs
from tokenizers import Tokenizer

from textattack.datasets import HuggingFaceDataset
from textattack.models.wrappers import ModelWrapper
import torch.nn.functional as F 

class DistilBertModelWrapper(ModelWrapper):
    def __init__(self, model_path):
        
        model_dict = torch.load(model_path)
        
        model_name = 'distilbert-base-cased'
        self.device = get_gpu_details()

        self.model = DistilBertForSequenceClassification.from_pretrained(model_name)
        self.model.load_state_dict(model_dict['distilbert'])
        self.model.eval()

        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
#         self.tokenizer.load_state_dict(model_dict['tokenizer'])
       
        if torch.cuda.is_available(): 
            self.model.cuda()
    
            if True:
                self.model = torch.nn.DataParallel(self.model)

    def __call__(self, test_inputs):
        
        # convert the text_inputs by passing into dataloader
        
        test_dataloader = [[],[]]
        
        for sent in test_inputs : 
        
            # encode the text
            test_dataloader[0].append((self.tokenizer.encode(sent, add_special_tokens=True, max_length=64, padding="max_length", truncation=True)))
        
            # attention
            test_dataloader[1].append(([int(token_id > 0) for token_id in test_dataloader[0][-1]]))
     
        #move to device GPU
        test_dataloader[0] = torch.tensor(test_dataloader[0])
        test_dataloader[1] = torch.tensor(test_dataloader[1])

        test_dataloader[0].to(self.device)
        test_dataloader[1].to(self.device)
                
        # do the forward pass
        with torch.no_grad():
            probs = F.softmax(self.model(test_dataloader[0], attention_mask=test_dataloader[1])['logits'], dim=1)
        return probs
            