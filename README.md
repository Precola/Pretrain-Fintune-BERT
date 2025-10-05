# Pretrain + Finetune BERT

## Pretrain
* run: `python ~/Tasks/TaskForPretraining.py`
* Data preprocess:
  * Paragraph -> Sentence
  * tokenizer
  * packing
  * datas_collator (for MLM)
* BERT-model:
  * BertEmbeddings()
  * BertEncoder
    * `nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])`
  * BertPooler
  * Head
    * BertForLMTransformHead    (MLM)
    * nn.Linear (NSP)

## Finetune
* run: `python dev_glue.py` (for Huggingface BERT)
* or run: `python dev_vanilla.py` (for yourself BERT)
Fix weights from word_embedding to the end of BERT encoder, initialize pooler weights.

When pretaining BERT for 400,000steps, finetuning on GLUE:
| System        | MNLI-(m/mm) | QQP  | QNLI | SST-2 | CoLA | STS-B | MRPC | RTE  | **Average** |
|----------------|-------------|------|------|--------|------|-------|------|------|-------------|
| **BERT<sub>LARGE</sub>** | 86.7 / 85.9 | 72.1 | 92.7 | 94.9 | – (60.5 acc) | 86.5 | 89.3 | 70.1 | 82.1 |
| **BERT<sub>ours</sub>**  | 82.1 / 82.6 | 87.3 | 89.8 | 90.9 | 53.3 | 86.3 | 89.7 | 55.2 | 79.7 |

**Table:** GLUE benchmark results.  
F1 scores are reported for QQP and MRPC, Spearman correlations for STS-B,  
Matthews correlation coefficient for CoLA, and accuracy scores for the other tasks.


## Reference
[transformers](https://github.com/huggingface/transformers/tree/main/src/transformers/models/bert)
[BertWithPretrained](https://github.com/moon-hotel/BertWithPretrained)