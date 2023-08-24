import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

# 아이디어 1: document embedding을 합친 다음에 classification model에 넣어서 분류하기
# 아이디어 2: classification model에서 나오는 연령 타겟에 대한 확률로 전체 자막 리스트의 각임베딩을 랭킹 매긴 다음에, 상위 n개를 뽑아서 classification model에 넣은 다음에 학습,
#           이 때 성능이 향상될 때마다 classification model weight을 복사하는 식으로 (knowledge distillation) 랭킹을 더 잘 매길 수 있게 함
#           중요한 점: 처음에 랜덤으로 n개를 넣을지, 아니면 document embedding으로 랭킹을 매긴 다음에 상위 n개를 넣을지
# https://aclanthology.org/2022.findings-emnlp.119.pdf -> 약간 이 논문처럼 class-wise ranking + class-wise-embed-rank similarity ranking이나 rule-based ranking을 하면 좋을 듯
# ranking loss를 추가해야 할지



labels = ["여성/가족", "남성", "성소수자", "인종/국적", "연령", "지역", "종교", "기타혐오", "악플/욕설", "clean"]
label_dict = {i: x for i, x in enumerate(labels)}

tokenizer = AutoTokenizer.from_pretrained('smilegate-ai/kor_unsmile')
model = AutoModel.from_pretrained('smilegate-ai/kor_unsmile')
classification_model = AutoModelForSequenceClassification.from_pretrained('smilegate-ai/kor_unsmile')

data = pd.read_csv("../data/LLM_infer/18세관람가/신세계.csv", encoding='utf-8')
prediction_list = [d for d in data['prediction']]

sentence_embeddings = []
for i in range(len(prediction_list)):
    test = prediction_list[i].strip().replace('\n', ' ')
    inputs = tokenizer(test, padding=True, truncation=True, return_tensors="pt", max_length=512)
    output = model(**inputs)
    # Mean pool the token-level embeddings to get sentence-level embeddings
    # Mean pool across the `sequence_length` dimension. Assumes that token embeddings have shape
    # `(batch_size, sequence_length, hidden_size)`
    embeddings = torch.sum(
        output.last_hidden_state * inputs['attention_mask'].unsqueeze(-1), dim=1
    ) / torch.clamp(torch.sum(inputs['attention_mask'], dim=1, keepdims=True), min=1e-9)
    print(embeddings.shape)
    sentence_embeddings.append(embeddings)

    # get cls embeddings
    # cls_embeddings = output.last_hidden_state[:, 0, :]
    # print(cls_embeddings.shape)

sentence_embeddings = torch.cat(sentence_embeddings, dim=0)
print(sentence_embeddings.shape)

predicted_probabilities = teacher_model.predict_proba(sentence_embeddings)
sorted_indeices = torch.argsort(predicted_probabilities, dim=0, descending=True)
sorted_embeddings = sentence_embeddings[sorted_indeices]

selected_embeddings = sorted_embeddings[:10]
student_model(selected_embeddings)


# output = classification_model(**inputs)
# logits = output[0]
# preds = logits.detach().cpu().numpy()
# label = np.argmax(preds)
# print(label_dict[label])