import openai
import re
import pandas as pd
import csv
import pandas as pd
from tqdm import tqdm

## api key
OPENAI_API_KEY="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"
openai.api_key=OPENAI_API_KEY

## read file
data = pd.read_csv('data/df_rate.tsv', sep='\t')
# csv_file=open('/home/jhlee/유해성/chatgpt/data/test.tsv')
# csv_reader=csv.reader(csv_file, delimiter=',')

## csv reader
texts =[]
labels = []

for text, label in zip(data['sentence'], data['label']):
    texts.append(text)
    labels.append(label)

## api call
model = "gpt-3.5-turbo"

# text+=text[0]
# key_facts+=key_facts[0]
# question+=question[0]
# issue+=issue[0]
# issue_solve+=issue_solve[0]

print(len(text))

ini=0

while True:
    iidx = []
    results = []

    for idx, (_, _) in enumerate(zip(texts[ini:], labels[ini:]), start=ini):
        #1 query = '법률 사례:' + text[idx+1]+'\n'+"사실 관계: "+'\n'+key_facts[idx+1]+'\n'+ '법률 사례' + text[idx]+'\n'+"사실 관계: "
        #2 query = '[사례]\n' + text[idx] + '\n[주요 사실 관계]\n' + key_facts[idx] + '\n[질문]\n' + question[idx] + '\n이에 대한 답변을 법률적 쟁점을 들어 항목별로 간단히 작성하시오'
        #3 query = '[사례]\n' + text[idx] + '\n[질문]\n' + question[idx] + '\n[주요 사실 관계]\n' + key_facts[idx] + '\n[질문]을 해결하기 위한 법률적 쟁점을 작성하고, 이에 대하여 서술하시오.'
        query = '[텍스트]\n' + texts[idx] + ' (텍스트를 읽고 유해성이 있으면 1, 없으면 0으로 답변하세요.)'

        messages = [
            {"role": "system", "content": "you are a professional data monitoring agent. Only answer harmness with either 0 or 1"},
            {"role": "user", "content": query}
        ]

        try:
            response = openai.ChatCompletion.create(model=model, messages=messages)
            answer = response['choices'][0]['message']['content']
            print("index: ", idx)
            # print(query)
            print("prediction: ", answer)
            print()
            iidx.append(idx)
            results.append({"id": idx, "prompt": query, "prediction": answer, 'label': labels[idx]})

        except:
            print(f"{idx}_error")
            break

    df = pd.DataFrame(results)
    df.to_csv('result/dev_result.tsv', sep='\t', index=False)

    ini=ini+len(iidx)
    if ini>=len(text):
        break