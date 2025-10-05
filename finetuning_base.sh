#!/bin/bash


source ~/zoe_env/bin/activate
cd /home/zoe/TTFS-BERT/Tasks/

#task_to_keys = {
#    "cola": ("sentence", None),
#    "mnli": ("premise", "hypothesis"),
#    "mrpc": ("sentence1", "sentence2"),
#    "qnli": ("question", "sentence"),
#    "qqp": ("question1", "question2"),
#    "rte": ("sentence1", "sentence2"),
#    "sst2": ("sentence", None),
#    "stsb": ("sentence1", "sentence2"),
#    "wnli": ("sentence1", "sentence2"),
#}
#python HFBERT.py  >> /home/zoe/TTFS-BERT/logs/output_base_HF_sst2.txt 2>&1
#python HFBERT.py  >> /home/zoe/TTFS-BERT/logs/output_base_HF_stsb.txt 2>&1

#python TaskForSingleSentenceClassification.py --task_name "cola" --learning_rate 2e-5 >> /home/zoe/TTFS-BERT/logs/output_base_cola.txt 2>&1
#python TaskForSingleSentenceClassification.py --task_name "mrpc" --learning_rate 2e-5> /home/zoe/TTFS-BERT/logs/output_base_mrpc.txt 2>&1
#python TaskForSingleSentenceClassification.py --task_name "rte" --learning_rate 2e-5>>  /home/zoe/TTFS-BERT/logs/output_base_rte.txt 2>&1
python TaskForSingleSentenceClassification.py --task_name "sst2" --learning_rate 3e-5> /home/zoe/TTFS-BERT/logs/output_base_sst2_.txt 2>&1
#python TaskForSingleSentenceClassification.py --task_name "stsb" --learning_rate 1e-5 >>  /home/zoe/TTFS-BERT/logs/output_base_stsb_vanilla.txt 2>&1
#python TaskForSingleSentenceClassification.py --task_name "wnli" --learning_rate 1e-5>> output_base_wnli_248179.txt 2>&1
#python TaskForSingleSentenceClassification.py --task_name "qnli" --learning_rate 3e-5> /home/zoe/TTFS-BERT/logs/output_base_qnli.txt 2>&1
#python TaskForSingleSentenceClassification.py --task_name "qqp" --learning_rate 3e-5  >> output_base_qqp_248179.txt 2>&1
#python TaskForSingleSentenceClassification.py --task_name "mnli" --learning_rate 3e-5 >> output_base_mnli_248179.txt 2>&1


#python TaskForSingleSentenceClassification_TTFS.py --seed 713 --task_name "rte" --learning_rate 3e-5>> output_rte.txt 2>&1
##python TaskForSingleSentenceClassification_TTFS.py --seed 1118 --task_name "cola">> output_qnli.txt 2>&1

#wait