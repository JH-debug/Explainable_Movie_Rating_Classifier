import torch
from torch.utils.data import Dataset
from collections import defaultdict
import numpy as np
import pandas as pd
from transformers.data.data_collator import DataCollatorForTokenClassification


class KoBigbirdMovieDataset(Dataset):
    def __init__(self, data_dir, tokenizer, max_length=4096, split='train', use='LLM_infer'):
        # self.data = defaultdict(list)
        # with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        #     reader = csv.DictReader(f)  #
        #     for row in reader:
        #         for (k, v) in row.items():
        #             self.data[k.strip()].append
        self.use = use
        if self.use == 'LLM_infer':
            self.data = pd.read_csv(data_dir + f'LLM_infer/LLM_{split}_concat.csv', encoding='utf-8')
        else:
            self.data = pd.read_csv(data_dir + f'kobigbird_{split}.csv', encoding='utf-8')
        self.tokenizer = tokenizer
        self.max_length = max_length


        self.labels = ['전체관람가', '12세관람가', '15세관람가', '18세관람가']
        self.aspects = ['주제', '선정성', '폭력성', '대사', '공포', '약물', '모방위험']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.use == 'LLM_infer':
            subtitles = self.data['LLM_infer'][idx]
        else:
            subtitles = self.data['Subtitle'][idx]
        # Tokenize the description
        inputs = self.tokenizer.encode_plus(
            subtitles,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        # Get the classification label
        rating = self.data['age'][idx]
        # Convert the rating to a class label
        classification_label = self.labels.index(rating.strip())

        # Get the explanation and Tokenize it
        # explanation = '이 영화의 내용에 따라 '
        # for aspect in self.aspects[:-1]:
        #     explanation += aspect + '에 대한 기준 등급은' + self.data[aspect][idx] + '이고, '
        # explanation += self.aspects[-1] + '에 대한 기준 등급은' + self.data[self.aspects[-1]][idx] + '입니다.'
        #
        # generation_label = self.tokenizer.encode(
        #     explanation,
        #     truncation=True,
        #     padding='max_length',
        #     max_length=self.max_length,
        #     return_tensors='pt'
        #
        return {'input_ids': inputs['input_ids'].squeeze(0),
                'attention_mask': inputs['attention_mask'].squeeze(0),
                'labels': classification_label}

    def collate_fn(self, batch):
        batch = self.tokenizer.pad(batch, return_tensors='pt')
        return batch


class KoT5SumInferDataset(Dataset):
    def __init__(self, data_dir, tokenizer, max_length=4096, split='train'):
        self.data = pd.read_csv(data_dir + f'sum_infer_{split}.csv', encoding='utf-8')
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.labels = ['전체관람가', '12세관람가', '15세관람가', '18세관람가']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sum_subtitles = self.data['Subtitle_sum'][idx]

        inputs = self.tokenizer.encode_plus(
            sum_subtitles,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        # Get the classification label
        rating = self.data['age'][idx]
        # Convert the rating to a class label
        classification_label = self.labels.index(rating.strip())

        return {'input_ids': inputs['input_ids'].squeeze(0),
                'attention_mask': inputs['attention_mask'].squeeze(0),
                'labels': classification_label}

    def collate_fn(self, batch):
        batch = self.tokenizer.pad(batch, return_tensors='pt')
        return batch


class LongKeT5Dataset(Dataset):
    def __init__(self, data_dir, tokenizer, max_length=4096, split='train'):
        self.data = pd.read_csv(data_dir + f'kobigbird_{split}.csv', encoding='utf-8')
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.labels = ['전체관람가', '12세관람가', '15세관람가', '18세관람가']
        self.aspects = ['주제', '선정성', '폭력성', '대사', '공포', '약물', '모방위험']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        subtitles = self.data['Subtitle'][idx]
        # Tokenize the description
        inputs = self.tokenizer.encode_plus(
            subtitles,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        # Get the classification label
        rating = self.data['age'][idx]
        # Convert the rating to a class label
        classification_label = self.tokenizer.encode_plus(
            rating,
            truncation=True,
            padding='max_length',
            max_length=5,
            return_tensors='pt'
        )

        return {'input_ids': inputs['input_ids'].squeeze(0),
                'attention_mask': inputs['attention_mask'].squeeze(0),
                'labels': classification_label['input_ids'].squeeze(0)}

    def collate_fn(self, batch):
        batch = self.tokenizer.pad(batch, return_tensors='pt')
        return batch


if __name__ == "__main__":
    from transformers import AutoTokenizer
    # tokenizer = AutoTokenizer.from_pretrained('monologg/kobigbird-bert-base')
    # dataset = KoBigbirdMovieDataset('data/', tokenizer)

    tokenizer = AutoTokenizer.from_pretrained('KETI-AIR/long-ke-t5-base')
    dataset = LongKeT5Dataset('data/', tokenizer)
    print(tokenizer.decode(dataset[0]['labels'], skip_special_tokens=True))
    print(dataset[0])