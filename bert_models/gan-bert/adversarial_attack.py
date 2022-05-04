# Quiet TensorFlow.
import os
import sys
sys.path.append('../utils')
from ganbert_utils import *
from ganbert_models import *

import numpy as np
# from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, pipeline

from textattack import Attacker
from tokenizers import Tokenizer
from textattack.attack_recipes import TextFoolerJin2019
from textattack.datasets import HuggingFaceDataset
from textattack.models.wrappers import ModelWrapper

class InferenceGANBert(nn.Module):
    
    def __init__(self, transformer, discriminator):
        super().__init__()
        self.transformer = transformer
        self.transformer.eval() 
        self.discriminator = discriminator
        self.discriminator.eval()
        
    def forward(self, dataloader, batch_size=64):
        # do the forward pass
        
        device = get_gpu_details()
        
        for batch in dataloader:
            # Unpack this training batch from our dataloader. 
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            with torch.no_grad():        
                model_outputs = self.transformer(b_input_ids, attention_mask=b_input_mask)
                hidden_states = model_outputs[-1]
                _, _, probs = self.discriminator(hidden_states)
                predicted_probs.extend(probs)
                
        return predicted_probs
            


class BertModelWrapper(ModelWrapper):
    """Transformers sentiment analysis pipeline returns a list of responses,
    like
        [{'label': 'POSITIVE', 'score': 0.7817379832267761}]
    We need to convert that to a format TextAttack understands, like
        [[0.218262017, 0.7817379832267761]
    """

    def __init__(self, model_path, batch_size=64):
        
        model_dict = torch.load(model_path)
        model_name = 'bert-base-cased'
        self.transformer = AutoModel.from_pretrained(model_name)
#         self.transformer.load_state_dict(model_dict['bert_encoder'])
        self.transformer.eval()
        print("type of transformer: ", type(self.transformer))
        
        self.discriminator = Discriminator()
        self.discriminator.load_state_dict(model_dict['discriminator'])
        self.discriminator.eval()
        print("type of discriminator: ", type(self.discriminator))
        
        self.tokenizer = Tokenizer()
        self.tokenizer.load_state_dict(model_dict['tokenizer'])
        
        self.model = InferenceGANBert(self.transformer, self.discriminator)
        self.batch_size = batch_size

    def __call__(self, test_inputs):
        
        # convert the text_inputs by passing into dataloader
        
        test_label_masks = np.ones(len(test_inputs), dtype=bool)

        test_dataloader = generate_data_loader(test_inputs, 
                                       test_label_masks, 
                                       {'0': 0, '1': 1}, 
                                       self.tokenizer, 
                                       batch_size=self.batch_size,
                                       do_shuffle = False)
        
        # unwrap the model contents and do the actual computation
        return self.model(test_dataloader, self.batch_size)
        
        


model_wrapper = BertModelWrapper(model_path='gan_bert_fine_tuned_6652.pt')

# Create the recipe
recipe = TextFoolerJin2019.build(model_wrapper)

dataset = HuggingFaceDataset("SetFit/sst2", split="test")

attacker = Attacker(recipe, dataset)
results = attacker.attack_dataset()