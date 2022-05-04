# ### Jupyter Notebook to Create Data Augmentations for Adversarial Training 

# We Plan to create 4 Different Augmented Datasets based on different recipes of making 
# 
# 1) EmbeddingAugmenter
# 
# 2) SynonymInsertionAugmenter
# 
# 3) WordNetAugmenter 
# 
# 4) BackTranslationAugmenter

import numpy as np
import pandas as pd
import torch
import transformers
import warnings
import time
import csv        

from textattack.augmentation.recipes import EmbeddingAugmenter, SynonymInsertionAugmenter,WordNetAugmenter,BackTranslationAugmenter

def get_sst_examples(input_file, test=False, discard_values = 0.5):

    train_examples = []
    test_examples = []

    with open(input_file, 'r') as f:

        contents = f.read()
        file_as_list = contents.splitlines()
        for line in file_as_list[1:]:
            
            is_dropped = np.random.binomial(1, discard_values, 1)[0]
            
            if not test and is_dropped == 1:
                continue
                
                
            text, label = line.split("\t") 
            if test:
                test_examples.append((text, label))
            else : 
                train_examples.append((text, label))
        f.close()

    return train_examples, test_examples


def generate_augmented_examples(input_examples,output_tsv,Augmenter , pct_words_to_swap=0.25 , transformations_per_example = 2):
    print(f"Length of Original Document - {len(input_examples)} \n")
    
    augmented_examples = [] 
    
    print(f"Initiating Creation of Data Augmentation\n")
    
    rng = np.random.default_rng() 
    
    augmented_indexes = rng.choice(len(input_examples), 20_000, replace=False)
    
    augmenter = Augmenter(pct_words_to_swap = pct_words_to_swap, transformations_per_example = transformations_per_example)
    
    for index in augmented_indexes : 
        
        augmented_strings = augmenter.augment(input_examples[index][0])
        
        augmented_examples += [(x,input_examples[index][1]) for x in augmented_strings]
        
        if len(augmented_examples) % 10 == 0 : 
            print(f"Generated {len(augmented_examples)} out of 20_000 Examples ", end = "\r")
    
    print(f"Data Generated , Writing it to Augmented Tab Separated Format {output_tsv} : ")
    
    with open(output_tsv, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        
        for examples in augmented_examples:
            writer.writerow(examples)
    
    print(f"All Output Written to {output_tsv}")

if __name__ == "__main__":

    labeled_examples, _ = get_sst_examples('./../../data/SST-2/train.tsv',test=False,discard_values = 0)

    generate_augmented_examples(labeled_examples,"SynonymInsertionAugmenter.tsv", SynonymInsertionAugmenter , pct_words_to_swap=0.2 , transformations_per_example = 2)


