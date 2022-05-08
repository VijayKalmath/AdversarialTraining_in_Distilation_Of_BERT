import sys 
sys.path.append('../utils')
from datetime import datetime 
from distilbert_utils import *
from distilbert_finetuning import * 
from transformers import AdamW
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

warnings.filterwarnings('ignore')

if __name__ == "__main__":

    GPU  = get_gpu()

    labeled_examples, _ = get_sst_examples('./../../data/SST-2/train.tsv',test=False,discard_values = 0)
    _, test_examples = get_sst_examples('./../../data/SST-2/dev.tsv', test=True,discard_values=0)

    print("\n\n SST Data Extracted and Read")
    print("Size of Training Data",len(labeled_examples))
    print("Size of Test Data", len(test_examples))


    label_map = {'0': 0, '1': 1}
    train_examples = labeled_examples


    transformer_type = 'distilbert-base-cased'

    tokenizer = DistilBertTokenizer.from_pretrained(transformer_type)

    print("\n\n")
    print("Generating DataSet from SST2")
    print("\n\n")
    train_dataloader = generate_data_loader(train_examples, label_map,tokenizer,batch_size =64, do_shuffle = True)

    test_dataloader = generate_data_loader(test_examples, label_map,tokenizer)

    #Create the tokenizer, model, optimizer, and criterion
    model = transfer_device(GPU, DistilBertForSequenceClassification.from_pretrained(transformer_type))
    
    epoch_number = 10 
    optimizer = AdamW(model.parameters(), lr=2e-5)
    criterion = binary_cross_entropy

    #Train and save the model
    print("\n\n\n\n")
    print("Intiating Training of DistilBert Model")
    print(f"Training Start Time : {datetime.now():%Y-%m-%d_%H-%M-%S%z}")
    print("\n\n\n")

    model = train_model(GPU, train_dataloader, test_dataloader, tokenizer, model, optimizer, criterion,epochs = epoch_number )
    
    
    print("\n\n")
    print(f"Training Completed at Time : {datetime.now():%Y-%m-%d_%H-%M-%S%z}")

    torch.save({
    'tokenizer': tokenizer,
    'distilbert': model.state_dict(),
}, f"distill_bert_finetuned_sst2_{len(train_examples)}_samples_{datetime.now():%Y-%m-%d_%H-%M-%S%z}.pt")
