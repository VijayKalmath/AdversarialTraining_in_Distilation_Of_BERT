import os
import sys
sys.path.append('../utils')

from distilbert_utils import *
from distilbert_finetuning import *

import numpy as np
from transformers import AutoTokenizer, DistilBertTokenizer, DistilBertForSequenceClassification,DistilBertConfig

from textattack import Attacker,AttackArgs
from tokenizers import Tokenizer

from textattack.datasets import HuggingFaceDataset
from textattack.models.wrappers import ModelWrapper
import torch.nn.functional as F 

from transformers import DistilBertModel
from  transformers.models.distilbert.modeling_distilbert import Transformer

def distill_finetune_weights(teacher, student):
    """
    Recursively copies the weights of the (teacher) to the (student).
    This function is meant to be first called on a RobertaFor... model, but is then called on every children of that model recursively.
    The only part that's not fully copied is the encoder, of which only half is copied.
    """

    if isinstance(teacher, DistilBertModel) or type(teacher).__name__.startswith('DistilBertForSequenceClassification'):

        for teacher_part, student_part in zip(teacher.children(), student.children()):
            distill_finetune_weights(teacher_part, student_part)

    # Else if the part is an encoder, copy one out of every layer
    elif isinstance(teacher, Transformer):

            teacher_encoding_layers = [layer for layer in next(teacher.children())]
            student_encoding_layers = [layer for layer in next(student.children())]

            for i in range(len(student_encoding_layers)):
                student_encoding_layers[i].load_state_dict(teacher_encoding_layers[2*i].state_dict())
    # Else the part is a head or something else, copy the state_dict
    else:

        student.load_state_dict(teacher.state_dict())

## Function
def distill_distilbert(teacher_model):
    """
    Distilates a RoBERTa (teacher_model) like would DistilBERT for a BERT model.
    The student model has the same configuration, except for the number of hidden layers, which is // by 2.
    The student layers are initilized by copying one out of two layers of the teacher, starting with layer 0.
    The head of the teacher is also copied.
    """
    # Get teacher configuration as a dictionary
    configuration = teacher_model.config.to_dict()
#     print(configuration)
    # Half the number of hidden layer
    configuration['n_layers'] //= 2
    # Convert the dictionnary to the student configuration
    configuration = DistilBertConfig.from_dict(configuration)
    # Create uninitialized student model
    student_model = type(teacher_model)(configuration)
    # Initialize the student's weights
    distill_finetune_weights(teacher=teacher_model, student=student_model)

    # Return the student model
    return student_model
        
class DistilBertModelWrapper(ModelWrapper):
    def __init__(self, model_path,isgan_distil=False):
        
        model_dict = torch.load(model_path)
        
        model_name = 'distilbert-base-cased'
        self.device = get_gpu_details()
        
        self.model = DistilBertForSequenceClassification.from_pretrained(model_name)
                
        if isgan_distil : 
            self.model  = distill_distilbert(self.model)
        
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
            
