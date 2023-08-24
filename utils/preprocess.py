import os
import pandas as pd

# Concat the subtitles of the same movie
def concat_subtitles():
    dataset = pd.read_csv('../data/df_rate.csv', encoding='utf-8')
    movie = {}

    for index, row in dataset.iterrows():
        movie_title = row['Movie_name']
        subtitle = row['Subtitle']
        label = row['age']

        if movie_title in movie:
            movie[movie_title]['subtitle'].append(subtitle)
            movie[movie_title]['label'].append(label)
        else:
            movie[movie_title] = {'subtitle': [subtitle], 'label': [label]}

    final_data = pd.DataFrame({'Movie_name': movie.keys(),
                               'Subtitle': list([movie[movie_title]['subtitle'] for movie_title in movie.keys()]),
                               'age': [movie[movie_title]['label'][0] for movie_title in movie.keys()]})
    final_data.to_csv('../data/df_concat.csv', index=False)

    print(len(movie))


# Concat the LLM infer dataset
def concat_LLM_infer():
    list_dir = [file for file in os.listdir('../data/LLM_infer') if not file.endswith('csv')]
    Movie_name = []
    Subtitle = []
    Prompt = []
    ages = []
    LLM_infer = []

    for age in list_dir:
        movie_list = [file for file in os.listdir(f'../data/LLM_infer/{age}') if file.endswith('csv')]
        for movie in movie_list:
            movie_data = pd.read_csv(f'../data/LLM_infer/{age}/{movie}', encoding='utf-8')
            for (subtitle, prompt, prediction) in zip(movie_data['subtitle'], movie_data['prompt'], movie_data['prediction']):
                Movie_name.append(movie.split('.csv')[0])
                Subtitle.append(subtitle)
                Prompt.append(prompt)
                LLM_infer.append(prediction)
                ages.append(age)

    LLM_concat_data = pd.DataFrame({'Movie_name': Movie_name,
                                'Subtitle': Subtitle,
                                'prompt': Prompt,
                                'LLM_infer': LLM_infer,
                                'age': ages})

    LLM_concat_data.to_csv('../data/LLM_infer/LLM_infer_concat.csv', index=False)
    print(len(LLM_concat_data))
    

if __name__ == "__main__":
    concat_LLM_infer()