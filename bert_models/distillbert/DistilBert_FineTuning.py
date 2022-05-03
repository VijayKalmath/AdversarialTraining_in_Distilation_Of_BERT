import sys 
sys.path.append('../utils')
from distilbert_utils import *
from distilbert_finetuning import * 
from transformers import AdamW
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

warnings.filterwarnings('ignore')

if __name__ == "__main__":

    GPU  = get_gpu()

    labeled_examples, _ = get_sst_examples('./../../data/SST-2/train.tsv',test=False,discard_values = 0.92)
    _, test_examples = get_sst_examples('./../../data/SST-2/dev.tsv', test=True,discard_values=1)

    len(labeled_examples), len(test_examples)


    label_map = {'0': 0, '1': 1}
    train_examples = labeled_examples


    transformer_type = 'distilbert-base-cased'

    tokenizer = DistilBertTokenizer.from_pretrained(transformer_type)

    train_dataloader = generate_data_loader(train_examples, label_map,tokenizer,batch_size =64, do_shuffle = True)

    test_dataloader = generate_data_loader(test_examples, label_map,tokenizer)

    #Create the tokenizer, model, optimizer, and criterion
    model = transfer_device(GPU, DistilBertForSequenceClassification.from_pretrained(transformer_type))

    optimizer = AdamW(model.parameters(), lr=2e-5)
    criterion = binary_cross_entropy

    #Train and save the model
    model = train_model(GPU, train_dataloader, test_dataloader, tokenizer, model, optimizer, criterion,epochs =1 )

