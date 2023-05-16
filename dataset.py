import torch
from torch.utils.data import Dataset
from collections import defaultdict
import csv


class MovieDataset(Dataset):
    def __init__(self, filename, tokenizer, max_length=512):
        self.data = defaultdict(list)
        with open(filename, 'r', encoding='cp949', errors='ignore') as f:
            reader = csv.DictReader(f)  #
            for row in reader:
                for (k, v) in row.items():
                    self.data[k.strip()].append(v)

        self.tokenizer = tokenizer
        self.max_length = max_length

        self.labels = ['전체관람가', '12세이상관람가', '15세이상관람가', '청소년관람불가']
        self.aspects = ['주제', '선정성', '폭력성', '대사', '공포', '약물', '모방위험']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Tokenize the description
        inputs = self.tokenizer.encode_plus(
            self.data['영화 요약'][idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        # Get the classification label
        rating = self.data['관람등급'][idx]
        # Convert the rating to a class label
        classification_label = self.labels.index(rating.strip())

        # Get the explanation and Tokenize it
        explanation = '이 영화의 내용에 따라 '
        for aspect in self.aspects[:-1]:
            explanation += aspect + '에 대한 기준 등급은' + self.data[aspect][idx] + '이고, '
        explanation += self.aspects[-1] + '에 대한 기준 등급은' + self.data[self.aspects[-1]][idx] + '입니다.'

        generation_label = self.tokenizer.encode(
            explanation,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'classification_label': torch.tensor(classification_label),
            'generation_label':generation_label.squeeze()
        }



if __name__ == "__main__":
    # import csv
    # from collections import defaultdict
    #
    # data = defaultdict(list)
    # with open('data/data.csv', 'r', encoding='cp949', errors='ignore') as f:
    #     reader = csv.DictReader(f)  #
    #     for row in reader:
    #         for (k, v) in row.items():
    #             data[k.strip()].append(v)
    #
    # explanation = '이 영화의 내용에 따라 '
    # aspects = ['주제', '선정성', '폭력성', '대사', '공포', '약물', '모방위험']
    # for aspect in aspects[:-1]:
    #     explanation += aspect + '에 대한 기준 등급은' + data[aspect][0] + '이고, '
    # explanation += aspects[-1] + '에 대한 기준 등급은' + data[aspects[-1]][0] + '입니다.'
    #
    # print(explanation)

    from transformers import BartTokenizer
    dataset = MovieDataset('data/data.csv', BartTokenizer.from_pretrained('hyunwoongko/kobart'))
    print(dataset[0])