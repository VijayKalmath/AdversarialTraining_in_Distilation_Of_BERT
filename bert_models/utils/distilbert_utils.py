#Libraries needed
import numpy as np
import pandas as pd

import torch
import transformers

import warnings
import time

from torch.utils.data import Dataset
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

#Get the GPU device if it exists, load the SST-2 dataset, and create PyTorch datasets and dataloaders for the training and validation sets
def get_gpu():
    #Check if a GPU is avaliable and if so return it
    GPU  =  None
    if torch.cuda.is_available():
        print("Using GPU")
        GPU  = torch.device("cuda")
    else:
        print("No GPU device avaliable! Using CPU")
    return  GPU

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

def get_sst_examples(input_file, test=False, discard_values = 0.5):

    train_examples = []
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
            else : 
                train_examples.append((text, label))
        f.close()

    return train_examples, test_examples

def generate_data_loader(input_examples, label_map,tokenizer,batch_size=64, do_shuffle = False, balance_label_examples = False):
    '''
    Generate a Dataloader given the input examples, eventually masked if they are 
    to be considered NOT labeled.
    '''

    #-----------------------------------------------
    # Generate input examples to the Transformer
    #-----------------------------------------------
    input_ids = []
    input_mask_array = []
    label_id_array = []

    # Tokenization 
    for text in input_examples:
        # each sentence is tokenized and converted into an ID from the vocabulary
        
        encoded_sent = tokenizer.encode(text[0], add_special_tokens=True, max_length=64, padding="max_length", truncation=True)
        
        input_ids.append(encoded_sent)
        label = [0]*2
        label[label_map[text[1]]] =  1
        label_id_array.append(label)
        
    # input_ids ---> contains a list of list of all word embeddings for the sentence 
    # label_id_array ---> contains a list of actual labels which can be (0, 1, 'UNK')
  
    # Attention to token (to ignore padded input wordpieces)
    for sent in input_ids:
        att_mask = [int(token_id > 0) for token_id in sent]                          
        input_mask_array.append(att_mask)
    # Convertion to Tensor
    
    input_ids = torch.tensor(input_ids) 
    input_mask_array = torch.tensor(input_mask_array)
    label_id_array = torch.tensor(label_id_array)# , dtype=torch.long)
    # Building the TensorDataset
    dataset = TensorDataset(input_ids, input_mask_array, label_id_array)
    
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
    return str(datetime.timedelta(seconds=elapsed_rounded))


def  get_gpu():
    #Check if a GPU is avaliable and if so return it
    GPU  =  None
    if torch.cuda.is_available():
        print("Using GPU")
        GPU  = torch.device("cuda")
    else:
        print("No GPU device avaliable! Using CPU")
    return  GPU

#Name: 		transfer_device
#Purpose: 	transfers model / data to the GPU devie if present
#Inputs: 	GPU -> GPU device if applicable, none if not
# 		 	data -> data to transfer
#Output: 	data -> data that has been transferred if applicable

def  transfer_device(GPU, data):
    if(GPU  !=  None):
        data = data.to(GPU)
    return data

#Name: 		count_correct
#Purpose: 	count the number of correct model predictions in a batch
#Inputs: 	predictions -> model predictions
#		 	targets -> target labels
#Outputs: 	correct -> number of correct model predictions
def  count_correct(predictions, targets):
	#Create variables to store the number of correct predictions along with the index of the prediction in the batch
    correct =  0
    index =  0
  
	#Loop across all predictions in the batch and count the number correct
    while(index <  len(predictions)):
        #Convert the prediction and target to lists
        prediction =  list(predictions[index])
        target =  list(targets[index])

        #Get the max index indicating the truth value from the prediction and target
        prediction_index = prediction.index(max(prediction))
        target_index = target.index(max(target))

        #If the max indices are the same increment correct
        if(prediction_index == target_index):
            correct +=  1
        index +=  1
    return correct


