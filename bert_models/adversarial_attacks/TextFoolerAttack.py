from ganbert_adversarial_attackutils import *

from textattack.attack_recipes import TextFoolerJin2019

def prepare_attack(model_path, source_model_name, num_adversarial_eg_test=10):
    
    model_wrapper = BertModelWrapper(model_path=model_path)
    sst2_dataset = HuggingFaceDataset("gpt3mix/sst2", split="train")
    
    # Create the recipe
    textfooler_recipe = TextFoolerJin2019.build(model_wrapper)
    
    attacker = Attacker(textfooler_recipe, sst2_dataset, 
                    AttackArgs(num_examples = num_adversarial_eg_test ,
                                   shuffle = True , # Shuffle Dataset 
                                   log_to_csv = f"GanBert_{source_model_name}_textfooler.csv" , # Log Attack to CSV 
                                   disable_stdout = True,# Supress individual Attack Results 
                              ))
    
    results = attacker.attack_dataset()

    
if __name__ == "__main__":
    source_model_name = "base"
    model_path = '../gan-bert/gan_bert_fine_tuned_6652.pt'
    num_adversarial_eg_test = 100
    prepare_attack(model_path, source_model_name, num_adversarial_eg_test)