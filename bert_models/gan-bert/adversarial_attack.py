# Quiet TensorFlow.
import os

import numpy as np
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, pipeline

from textattack import Attacker
from textattack.attack_recipes import PWWSRen2019
from textattack.datasets import HuggingFaceDataset
from textattack.models.wrappers import ModelWrapper


class BertModelWrapper(ModelWrapper):
    """Transformers sentiment analysis pipeline returns a list of responses,
    like
        [{'label': 'POSITIVE', 'score': 0.7817379832267761}]
    We need to convert that to a format TextAttack understands, like
        [[0.218262017, 0.7817379832267761]
    """

    def __init__(self, model_path, batch_size=64):
        model_dict = torch.load(model_path)
        self.transformer = model_dict['bert_encoder']
        self.discriminator = model_dict['discriminator']
        self.tokenizer = model_dict['tokenizer']
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
        self.transformer.eval() 
        self.discriminator.eval()
        
        
        # Tracking variables 
        predicted_probs = []

        
        # Evaluate data for one epoch
        
        for batch in test_dataloader:
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

# Create the model: a French sentiment analysis model.
# see https://github.com/TheophileBlard/french-sentiment-analysis-with-bert
model = TFAutoModelForSequenceClassification.from_pretrained("tblard/tf-allocine")
tokenizer = AutoTokenizer.from_pretrained("tblard/tf-allocine")
pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

model_wrapper = HuggingFaceSentimentAnalysisPipelineWrapper(pipeline)

# Create the recipe: PWWS uses a WordNet transformation.
recipe = PWWSRen2019.build(model_wrapper)
# WordNet defaults to english. Set the default language to French ('fra')
#
# See
# "Building a free French wordnet from multilingual resources",
# E. L. R. A. (ELRA) (ed.),
# Proceedings of the Sixth International Language Resources and Evaluation (LREC’08).

recipe.transformation.language = "fra"

dataset = HuggingFaceDataset("allocine", split="test")

attacker = Attacker(recipe, dataset)
results = attacker.attack_dataset()