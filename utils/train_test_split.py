import os
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split


def divide_baseline_dataset():
    if 'kobigbird_data.csv' not in os.listdir('../data'):
        data = pd.read_csv('../data/df_rate.csv', encoding='utf-8')
        data = data.loc[lambda x : x['Movie_name'] != '램페이지(12세)']
        movies = defaultdict(list)
        ages = []
        for i, (movie_name, subtitle, age) in enumerate(zip(data['Movie_name'], data['Subtitle'], data['age'])):
            if movie_name not in movies.keys():
                ages.append(age)
                movies[movie_name] = str(subtitle.strip())
            else:
                movies[movie_name] += ' ' + str(subtitle.strip())

        final_data = pd.DataFrame({'Movie_name': movies.keys(), 'Subtitle': list(movies.values()), 'age': ages})
        final_data.to_csv('data/kobigbird_data.csv', index=False)

    data = pd.read_csv('../data/kobigbird_data.csv', encoding='utf-8')
    train, test = train_test_split(data, test_size=0.2, random_state=42, stratify=data['age'])
    val, test = train_test_split(test, test_size=0.5, random_state=42, stratify=test['age'])

    print("train: ", len(train), "val: ", len(val), "test: ", len(test))

    train.to_csv('../data/kobigbird_train.csv', index=False)
    val.to_csv('../data/kobigbird_val.csv', index=False)
    test.to_csv('../data/kobigbird_test.csv', index=False)

    print(train['age'].value_counts())
    print(val['age'].value_counts())
    print(test['age'].value_counts())


def extract_movie_list():
    for data in ('train', 'val', 'test'):
        dataset = pd.read_csv(f'../data/kobigbird_{data}.csv', encoding='utf-8')
        with open(f'../data/{data}_영화_리스트.txt', 'w', encoding = 'utf-8') as f:
            for movie in dataset['Movie_name'].unique():
                f.write(movie + '\n')


def divide_LLM_infer_dataset():
    train_movie_list = [open('../data/train_영화_리스트.txt', 'r', encoding='utf-8').readlines()]
    train_movie_list = [movie.strip() for movie in train_movie_list[0]]
    val_movie_list = [open('../data/val_영화_리스트.txt', 'r', encoding='utf-8').readlines()]
    val_movie_list = [movie.strip() for movie in val_movie_list[0]]
    test_movie_list = [open('../data/test_영화_리스트.txt', 'r', encoding='utf-8').readlines()]
    test_movie_list = [movie.strip() for movie in test_movie_list[0]]

    dataset = pd.read_csv('../data/LLM_infer/LLM_infer_concat.csv', encoding='utf-8')
    train_dataset = dataset.loc[lambda x : x['Movie_name'].isin(train_movie_list)]
    val_dataset = dataset.loc[lambda x : x['Movie_name'].isin(val_movie_list)]
    test_dataset = dataset.loc[lambda x : x['Movie_name'].isin(test_movie_list)]

    train_dataset.to_csv('../data/LLM_infer/LLM_infer_train.csv', index=False)
    val_dataset.to_csv('../data/LLM_infer/LLM_infer_val.csv', index=False)
    test_dataset.to_csv('../data/LLM_infer/LLM_infer_test.csv', index=False)


if __name__ == "__main__":
    divide_LLM_infer_dataset()
    # extract_movie_list()