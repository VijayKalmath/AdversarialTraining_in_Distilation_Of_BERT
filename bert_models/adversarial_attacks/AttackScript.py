from distilbert_adversarial_attackutils import * 
from ganbert_adversarial_attackutils import *
import pickle
import sys 
import argparse

from textattack.attack_recipes import TextFoolerJin2019,Pruthi2019,DeepWordBugGao2018,BAEGarg2019
from textattack.datasets import Dataset 

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
                test_examples.append((text, int(label)))
            else : 
                train_examples.append((text, int(label)))
        f.close()
        
    return train_examples, test_examples


def prepare_attack(sst2_dataset,recipe_type,bert_model_type, model_path, source_model_name,isgan_distil, num_adversarial_eg_test=10):
    
    if bert_model_type == 'gan-bert':
        model_wrapper = BertModelWrapper(model_path=model_path)
    else:
        # put the new model wrapper here
        model_wrapper = DistilBertModelWrapper(model_path=model_path,isgan_distil = isgan_distil)
        
#     sst2_dataset = HuggingFaceDataset("gpt3mix/sst2", split="train")
    
    # Create the recipe
    if recipe_type == "TextFooler" :
        recipe = TextFoolerJin2019.build(model_wrapper)
    elif recipe_type == "BAE" :
        recipe = BAEGarg2019.build(model_wrapper)
    elif recipe_type == "DeepWordBug" :
        recipe = DeepWordBugGao2018.build(model_wrapper)
    else :
         recipe =  Pruthi2019.build(model_wrapper)
    
    
    attacker = Attacker(recipe, sst2_dataset, 
                    AttackArgs(num_examples = num_adversarial_eg_test ,
                                   shuffle = True , # Shuffle Dataset 
                                   log_to_csv = f"Attack_logs/{recipe_type}/{source_model_name}_{recipe_type}.csv" , # Log Attack to CSV 
                                   disable_stdout = True,# Supress individual Attack Results 
                              ))
    
    results = attacker.attack_dataset()
#     pickle.dump(results, open(f"Attack_logs/{recipe}/{source_model_name}_{recipe}_results.pickle", 'wb'))

    
if __name__ == "__main__":
  

    
    parser = argparse.ArgumentParser(description='Adversarial Attacks on Models')
    
    parser.add_argument('--bert_model_type', choices=['distilbert', 'gan-bert'])
    
    parser.add_argument('--text_attack_recipe', choices=['TextFooler', 'BAE','DeepWordBug','Pruthi'])    

    parser.add_argument('--isgan_distil',type=str,default="No")
    
    parser.add_argument('--model_path',  type=str)
    
    args = parser.parse_args()

    bert_model_type = args.bert_model_type

    textattack_recipe = args.text_attack_recipe

    model_path = args.model_path

    source_model_name = model_path.split('/')[-1][:-3]
    
    if args.isgan_distil == "No":
        isgandistil = False 
    else : 
        isgandistil = True
    
    print("="*40)
    print(args)
    
    
    _, test_examples = get_sst_examples('./../../data/SST-2/dev.tsv', test=True,discard_values = 0)    
    
    dataset = Dataset(test_examples)

    num_adversarial_eg_test = 1000
    
    prepare_attack(dataset,textattack_recipe,bert_model_type, model_path, source_model_name,isgandistil, num_adversarial_eg_test)
    print("$"*40 + "\n\n\n\n\n")
    
    