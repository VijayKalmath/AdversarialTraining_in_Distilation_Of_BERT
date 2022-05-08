import torch
import io
import random
import time
import math
import torch.nn.functional as F
from datetime import timedelta
import numpy as np
import torch.nn as nn
from transformers import *
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


def get_gpu_details():
    # If there's a GPU available...
    if torch.cuda.is_available():    
        # Tell PyTorch to use the GPU.    
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
        print('\n\n')
    # If not...
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
    return device 

def get_sst_examples(input_file, test=False, discard_values = 0.5,unknown_label_percentage=0.5):
    """Creates examples for the training and dev sets."""
    labeled_examples = []
    unlabeled_examples = []
    test_examples = []

    with open(input_file, 'r') as f:
        contents = f.read()
        file_as_list = contents.splitlines()
        for line in file_as_list[1:]:
            
            # random drop 90% of examples for checking
            is_dropped = np.random.binomial(1, discard_values, 1)[0]
            if not test and is_dropped == 1:
                continue
            
            text, label = line.split("\t") 
            if test:
                test_examples.append((text, label))
            else:
                if np.random.binomial(1, unknown_label_percentage, 1)[0] == 0:
                    unlabeled_examples.append((text, 'UNK'))
                else:
                    labeled_examples.append((text, label))
        f.close()

    return labeled_examples, unlabeled_examples, test_examples


def generate_data_loader(input_examples, label_masks, label_map, tokenizer, batch_size=64, do_shuffle = False):
    '''
    Generate a Dataloader given the input examples, eventually masked if they are 
    to be considered NOT labeled.
    '''
    examples = []

    # Count the percentage of labeled examples  
    num_labeled_examples = 0
    for label_mask in label_masks:
        if label_mask: 
            num_labeled_examples += 1
    label_mask_rate = num_labeled_examples/len(input_examples)
    
    ######################################################################################################################
    
    ##########  very trivial implementation of maintaining balance, not necesary for our case #############################

    # if required it applies the balance  
    for index, ex in enumerate(input_examples): 
        examples.append((ex, label_masks[index]))
                
    print("examples len is: ", len(examples))
  
    ######################################################################################################################
    
    #-----------------------------------------------
    # Generate input examples to the Transformer
    #-----------------------------------------------
    input_ids = []
    input_mask_array = []
    label_mask_array = []
    label_id_array = []

    # Tokenization 
    for (text, label_mask) in examples:
        # each sentence is tokenized and converted into an ID from the vocabulary
        encoded_sent = tokenizer.encode(text[0], add_special_tokens=True, max_length=64, padding="max_length", truncation=True)
        input_ids.append(encoded_sent)
        label_id_array.append(label_map[text[1]])
        label_mask_array.append(label_mask)
    
    # input_ids ---> contains a list of list of all word embeddings for the sentence 
    # label_id_array ---> contains a list of actual labels which can be (0, 1, 'UNK')
    # label_mask_array ---> contains a list of all label_mask that indicates whether dataset is labeled or not
    # input_mask_array ---> contains booleans to ignore the padded words
  
    # Attention to token (to ignore padded input wordpieces)
    for sent in input_ids:
        att_mask = [int(token_id > 0) for token_id in sent]                          
        input_mask_array.append(att_mask)
    # Convertion to Tensor
    input_ids = torch.tensor(input_ids) 
    input_mask_array = torch.tensor(input_mask_array)
    label_id_array = torch.tensor(label_id_array, dtype=torch.long)
    label_mask_array = torch.tensor(label_mask_array)

    # Building the TensorDataset
    dataset = TensorDataset(input_ids, input_mask_array, label_id_array, label_mask_array)

    if do_shuffle:
        sampler = RandomSampler
    else:
        sampler = SequentialSampler

    # Building the DataLoader
    return DataLoader(
                dataset,  # The training samples.
                sampler = sampler(dataset), 
                batch_size = batch_size) # Trains with this batch size.

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(timedelta(seconds=elapsed_rounded))

