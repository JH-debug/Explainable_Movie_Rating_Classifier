import pandas as pd
import numpy as np
from transformers import logging, AutoTokenizer, AutoModel
from torchmetrics.functional.text.bert import bert_score
from torchmetrics.text import BLEUScore, SacreBLEUScore
from torchmetrics.text.rouge import ROUGEScore

result = pd.read_csv('result/koat.csv')
print(result[result.isna().any(axis=1)])
result = result.dropna(axis=0)

bert_scores = bert_score(result['pre_output'].tolist(), result['real_output'].tolist(),
                         model_name_or_path = "klue/bert-base", verbose = True)

bleu_score = BLEUScore(n_gram=1, smooth=True)
sacre_bleu = SacreBLEUScore(n_gram=1, smooth=True)
rouge_score = ROUGEScore(tokenizer=AutoTokenizer.from_pretrained("klue/bert-base"))

print("bert_score: ")
for key in bert_scores.keys():
    print(key, np.mean(bert_scores[key]))

bleu_score_list = []
sacre_bleu_list = []

for pred, real in zip(result['pre_output'].tolist(), result['real_output'].tolist()):
    bleu_score_list.append(bleu_score(pred, real))
    sacre_bleu_list.append(sacre_bleu(pred, real))
    rouge_score.update(pred, real)

print("bleu_score: ", np.mean(bleu_score_list))
print("sacre_bleu: ", np.mean(sacre_bleu_list))
print("rouge_score: ", rouge_score.compute())