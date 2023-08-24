from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import pandas as pd

model_path = "KETI-AIR-Downstream/long-ke-t5-base-summarization"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
model.to('cuda:2')

prefix = "summarization-num_lines-{}: "

for data_type in ['train', 'val', 'test']:
    articles = pd.read_csv('/home/jhlee/Explainable_Movie_Rating_Classifier/data/kobigbird_{}.csv'.format(data_type),
                           encoding='utf-8')
    summarization_result = []

    for i in tqdm(range(len(articles['Subtitle']))):
        movie = articles.iloc[i]['Movie_name']
        article = articles.iloc[i]['Subtitle']
        age = articles.iloc[i]['age']
        input_text = prefix.format(5) + article
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids
        input_ids = input_ids.to('cuda:2')
        gen_seq = model.generate(
            input_ids,
            max_length=512
        )

        result = tokenizer.decode(gen_seq[0], skip_special_tokens=True)
        summarization_result.append(result)

    data = pd.DataFrame({'Movie_name': articles['Movie_name'], 'Subtitle_sum': summarization_result, 'age': articles['age']})
    data.to_csv("/home/jhlee/Explainable_Movie_Rating_Classifier/data/keT5_summarization/sum_infer_{}.csv".format(data_type),
                index=False)
