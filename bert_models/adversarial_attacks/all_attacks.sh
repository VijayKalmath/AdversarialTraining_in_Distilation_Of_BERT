#/bin/bash 

echo "Attacking Models with TextFooler"

python -u  AttackScript.py --bert_model_type gan-bert --text_attack_recipe TextFooler --model_path "../gan-bert/finetuned_models/gan_bert_finetuned_sst2_67349_samples_0.5_labelratio_2022-05-04_00-59-36.pt"

python -u  AttackScript.py --bert_model_type distilbert --text_attack_recipe TextFooler --model_path "../distilbert/finetuned_models/distill_bert_EmbeddingAugmenter_adversarialtrained_sst2_106494_samples_2022-05-05_18-40-16.pt"
python -u  AttackScript.py --bert_model_type distilbert --text_attack_recipe TextFooler --model_path "../distilbert/finetuned_models/distill_bert_SynonymInsertionAugmenter_adversarialtrained_sst2_106868_samples_2022-05-05_19-10-58.pt"
python -u  AttackScript.py --bert_model_type distilbert --text_attack_recipe TextFooler --model_path "../distilbert/finetuned_models/distill_bert_WordNetAugmenter_adversarialtrained_sst2_106748_samples_2022-05-05_20-18-41.pt"
python -u  AttackScript.py --bert_model_type distilbert --text_attack_recipe TextFooler --model_path "../distilbert/finetuned_models/distill_bert_finetuned_sst2_67349_samples_2022-05-03_21-30-41.pt"

python -u  AttackScript.py --bert_model_type distilbert --text_attack_recipe TextFooler --model_path "../distilbert/finetuned_models/distill_bert_fromsavedmodel_All_adversarialtrained_sst2_118063_samples_2022-05-07_17-01-42.pt"
python -u  AttackScript.py --bert_model_type distilbert --text_attack_recipe TextFooler --model_path "../distilbert/finetuned_models/distill_bert_fromsavedmodel_EmbeddingAugmenter_adversarialtrained_sst2_39145_samples_2022-05-07_16-27-25.pt"
python -u  AttackScript.py --bert_model_type distilbert --text_attack_recipe TextFooler --model_path "../distilbert/finetuned_models/distill_bert_fromsavedmodel_SynonymInsertionAugmenter_adversarialtrained_sst2_39519_samples_2022-05-07_16-25-27.pt"
python -u  AttackScript.py --bert_model_type distilbert --text_attack_recipe TextFooler --model_path "../distilbert/finetuned_models/distill_bert_fromsavedmodel_WordNetAugmenter_adversarialtrained_sst2_39399_samples_2022-05-07_00-47-45.pt"

python -u  AttackScript.py --bert_model_type distilbert --text_attack_recipe TextFooler --model_path "../bertmodel_distillation/Gan_distilbert_sst2_2022-05-08_03-47-09.pt" --isgan_distil Yes






echo "Attacking Models with BAE"



python -u  AttackScript.py --bert_model_type gan-bert --text_attack_recipe BAE --model_path "../gan-bert/finetuned_models/gan_bert_finetuned_sst2_67349_samples_0.5_labelratio_2022-05-04_00-59-36.pt"

python -u  AttackScript.py --bert_model_type distilbert --text_attack_recipe BAE --model_path "../distilbert/finetuned_models/distill_bert_EmbeddingAugmenter_adversarialtrained_sst2_106494_samples_2022-05-05_18-40-16.pt"
python -u  AttackScript.py --bert_model_type distilbert --text_attack_recipe BAE --model_path "../distilbert/finetuned_models/distill_bert_SynonymInsertionAugmenter_adversarialtrained_sst2_106868_samples_2022-05-05_19-10-58.pt"
python -u  AttackScript.py --bert_model_type distilbert --text_attack_recipe BAE --model_path "../distilbert/finetuned_models/distill_bert_WordNetAugmenter_adversarialtrained_sst2_106748_samples_2022-05-05_20-18-41.pt"
python -u  AttackScript.py --bert_model_type distilbert --text_attack_recipe BAE --model_path "../distilbert/finetuned_models/distill_bert_finetuned_sst2_67349_samples_2022-05-03_21-30-41.pt"

python -u  AttackScript.py --bert_model_type distilbert --text_attack_recipe BAE --model_path "../distilbert/finetuned_models/distill_bert_fromsavedmodel_All_adversarialtrained_sst2_118063_samples_2022-05-07_17-01-42.pt"
python -u  AttackScript.py --bert_model_type distilbert --text_attack_recipe BAE --model_path "../distilbert/finetuned_models/distill_bert_fromsavedmodel_EmbeddingAugmenter_adversarialtrained_sst2_39145_samples_2022-05-07_16-27-25.pt"
python -u  AttackScript.py --bert_model_type distilbert --text_attack_recipe BAE --model_path "../distilbert/finetuned_models/distill_bert_fromsavedmodel_SynonymInsertionAugmenter_adversarialtrained_sst2_39519_samples_2022-05-07_16-25-27.pt"
python -u  AttackScript.py --bert_model_type distilbert --text_attack_recipe BAE --model_path "../distilbert/finetuned_models/distill_bert_fromsavedmodel_WordNetAugmenter_adversarialtrained_sst2_39399_samples_2022-05-07_00-47-45.pt"

python -u  AttackScript.py --bert_model_type distilbert --text_attack_recipe BAE --model_path "../bertmodel_distillation/Gan_distilbert_sst2_2022-05-08_03-47-09.pt" --isgan_distil Yes







echo "Attacking Models with DeepWordBug"

python -u  AttackScript.py --bert_model_type gan-bert --text_attack_recipe DeepWordBug --model_path "../gan-bert/finetuned_models/gan_bert_finetuned_sst2_67349_samples_0.5_labelratio_2022-05-04_00-59-36.pt"

python -u  AttackScript.py --bert_model_type distilbert --text_attack_recipe DeepWordBug --model_path "../distilbert/finetuned_models/distill_bert_EmbeddingAugmenter_adversarialtrained_sst2_106494_samples_2022-05-05_18-40-16.pt"
python -u  AttackScript.py --bert_model_type distilbert --text_attack_recipe DeepWordBug --model_path "../distilbert/finetuned_models/distill_bert_SynonymInsertionAugmenter_adversarialtrained_sst2_106868_samples_2022-05-05_19-10-58.pt"
python -u  AttackScript.py --bert_model_type distilbert --text_attack_recipe DeepWordBug --model_path "../distilbert/finetuned_models/distill_bert_WordNetAugmenter_adversarialtrained_sst2_106748_samples_2022-05-05_20-18-41.pt"
python -u  AttackScript.py --bert_model_type distilbert --text_attack_recipe DeepWordBug --model_path "../distilbert/finetuned_models/distill_bert_finetuned_sst2_67349_samples_2022-05-03_21-30-41.pt"

python -u  AttackScript.py --bert_model_type distilbert --text_attack_recipe DeepWordBug --model_path "../distilbert/finetuned_models/distill_bert_fromsavedmodel_All_adversarialtrained_sst2_118063_samples_2022-05-07_17-01-42.pt"
python -u  AttackScript.py --bert_model_type distilbert --text_attack_recipe DeepWordBug --model_path "../distilbert/finetuned_models/distill_bert_fromsavedmodel_EmbeddingAugmenter_adversarialtrained_sst2_39145_samples_2022-05-07_16-27-25.pt"
python -u  AttackScript.py --bert_model_type distilbert --text_attack_recipe DeepWordBug --model_path "../distilbert/finetuned_models/distill_bert_fromsavedmodel_SynonymInsertionAugmenter_adversarialtrained_sst2_39519_samples_2022-05-07_16-25-27.pt"
python -u  AttackScript.py --bert_model_type distilbert --text_attack_recipe DeepWordBug --model_path "../distilbert/finetuned_models/distill_bert_fromsavedmodel_WordNetAugmenter_adversarialtrained_sst2_39399_samples_2022-05-07_00-47-45.pt"

python -u  AttackScript.py --bert_model_type distilbert --text_attack_recipe DeepWordBug --model_path "../bertmodel_distillation/Gan_distilbert_sst2_2022-05-08_03-47-09.pt" --isgan_distil Yes








echo "Attacking Models with Pruthi"



python -u  AttackScript.py --bert_model_type gan-bert --text_attack_recipe Pruthi --model_path "../gan-bert/finetuned_models/gan_bert_finetuned_sst2_67349_samples_0.5_labelratio_2022-05-04_00-59-36.pt"

python -u  AttackScript.py --bert_model_type distilbert --text_attack_recipe Pruthi --model_path "../distilbert/finetuned_models/distill_bert_EmbeddingAugmenter_adversarialtrained_sst2_106494_samples_2022-05-05_18-40-16.pt"
python -u  AttackScript.py --bert_model_type distilbert --text_attack_recipe Pruthi --model_path "../distilbert/finetuned_models/distill_bert_SynonymInsertionAugmenter_adversarialtrained_sst2_106868_samples_2022-05-05_19-10-58.pt"
python -u  AttackScript.py --bert_model_type distilbert --text_attack_recipe Pruthi --model_path "../distilbert/finetuned_models/distill_bert_WordNetAugmenter_adversarialtrained_sst2_106748_samples_2022-05-05_20-18-41.pt"
python -u  AttackScript.py --bert_model_type distilbert --text_attack_recipe Pruthi --model_path "../distilbert/finetuned_models/distill_bert_finetuned_sst2_67349_samples_2022-05-03_21-30-41.pt"

python -u  AttackScript.py --bert_model_type distilbert --text_attack_recipe Pruthi --model_path "../distilbert/finetuned_models/distill_bert_fromsavedmodel_All_adversarialtrained_sst2_118063_samples_2022-05-07_17-01-42.pt"
python -u  AttackScript.py --bert_model_type distilbert --text_attack_recipe Pruthi --model_path "../distilbert/finetuned_models/distill_bert_fromsavedmodel_EmbeddingAugmenter_adversarialtrained_sst2_39145_samples_2022-05-07_16-27-25.pt"
python -u  AttackScript.py --bert_model_type distilbert --text_attack_recipe Pruthi --model_path "../distilbert/finetuned_models/distill_bert_fromsavedmodel_SynonymInsertionAugmenter_adversarialtrained_sst2_39519_samples_2022-05-07_16-25-27.pt"
python -u  AttackScript.py --bert_model_type distilbert --text_attack_recipe Pruthi --model_path "../distilbert/finetuned_models/distill_bert_fromsavedmodel_WordNetAugmenter_adversarialtrained_sst2_39399_samples_2022-05-07_00-47-45.pt"

python -u  AttackScript.py --bert_model_type distilbert --text_attack_recipe Pruthi --model_path "../bertmodel_distillation/Gan_distilbert_sst2_2022-05-08_03-47-09.pt" --isgan_distil Yes




