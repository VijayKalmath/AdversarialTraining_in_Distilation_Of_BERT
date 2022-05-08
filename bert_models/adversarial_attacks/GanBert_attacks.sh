i#/bin/bash 

echo "Attacking Models with TextFooler"

python -u  AttackScript.py --bert_model_type gan-bert --text_attack_recipe TextFooler --model_path "../gan-bert/finetuned_models/gan_bert_finetuned_sst2_67349_samples_0.2_labelratio_2022-05-08_08-34-28.pt"
python -u  AttackScript.py --bert_model_type gan-bert --text_attack_recipe TextFooler --model_path "../gan-bert/finetuned_models/gan_bert_finetuned_sst2_67349_samples_0.7_labelratio_2022-05-08_15-18-12.pt"

echo "Attacking Models with BAE"



python -u  AttackScript.py --bert_model_type gan-bert --text_attack_recipe BAE --model_path "../gan-bert/finetuned_models/gan_bert_finetuned_sst2_67349_samples_0.2_labelratio_2022-05-08_08-34-28.pt"
python -u  AttackScript.py --bert_model_type gan-bert --text_attack_recipe BAE --model_path "../gan-bert/finetuned_models/gan_bert_finetuned_sst2_67349_samples_0.7_labelratio_2022-05-08_15-18-12.pt"




echo "Attacking Models with DeepWordBug"

python -u  AttackScript.py --bert_model_type gan-bert --text_attack_recipe DeepWordBug --model_path "../gan-bert/finetuned_models/gan_bert_finetuned_sst2_67349_samples_0.2_labelratio_2022-05-08_08-34-28.pt"
python -u  AttackScript.py --bert_model_type gan-bert --text_attack_recipe DeepWordBug --model_path "../gan-bert/finetuned_models/gan_bert_finetuned_sst2_67349_samples_0.7_labelratio_2022-05-08_15-18-12.pt"





echo "Attacking Models with Pruthi"



python -u  AttackScript.py --bert_model_type gan-bert --text_attack_recipe Pruthi --model_path "../gan-bert/finetuned_models/gan_bert_finetuned_sst2_67349_samples_0.2_labelratio_2022-05-08_08-34-28.pt"
python -u  AttackScript.py --bert_model_type gan-bert --text_attack_recipe Pruthi --model_path "../gan-bert/finetuned_models/gan_bert_finetuned_sst2_67349_samples_0.7_labelratio_2022-05-08_15-18-12.pt"
