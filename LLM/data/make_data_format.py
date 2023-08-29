import pandas as pd
import json


def make_train_test_dataset():
    train = pd.read_csv('../../data/LLM_infer/LLM_infer_train.csv')
    val = pd.read_csv('../../data/LLM_infer/LLM_infer_val.csv')
    trainval = pd.concat([train, val], axis=0)
    data = []
    with open('llm_train.json', 'w', encoding='utf-8') as make_file:
        for subtitle, infer in zip(trainval['Subtitle'], trainval['LLM_infer']):
            prompt = "유해성기준-{욕설, 비속어, 저속어: 저속한 욕설, 비속어, 저속어 표현}," +\
                     "{언어 습관이 미치는 영향: 공격적이고 수치심을 느끼게 하는 거친 표현이 있으나 내용 전개상 수용가능한 표현}," +\
                     "{차별적 인권침해적 언어 사용의 표현: 자극적이고 혐오스러운 성적 표현이나 정서적 인격적인 모욕감이나 수치심을 유발하는 표현} " + \
                     "유해성기준을 기반으로 대화의 유해성 여부를 판단하고 관련 대사를 출력하라. 유해성 대사가 없다면 없음 이라고 출력하라. " + \
                     "설명란에 유해성 판단의 근거 및 관련 줄거리를 총 125자 이내로 간결하게 요약하라."

            dict = {"instruction": prompt,
                    "input": subtitle.replace("['", '').replace("']", '').replace("'", ''),
                    "output": str(infer).replace("'", '"').replace('\"', '')}
            data.append(dict)
        json.dump(data, make_file, ensure_ascii=False, indent=4)


    test = pd.read_csv('../../data/LLM_infer/LLM_infer_test.csv')
    data = []
    with open('llm_test.json', 'w', encoding='utf-8') as make_file:
        for subtitle, infer in zip(test['Subtitle'], test['LLM_infer']):
            prompt = "유해성기준-{욕설, 비속어, 저속어: 저속한 욕설, 비속어, 저속어 표현}," +\
                     "{언어 습관이 미치는 영향: 공격적이고 수치심을 느끼게 하는 거친 표현이 있으나 내용 전개상 수용가능한 표현}," +\
                     "{차별적 인권침해적 언어 사용의 표현: 자극적이고 혐오스러운 성적 표현이나 정서적 인격적인 모욕감이나 수치심을 유발하는 표현} " + \
                     "유해성기준을 기반으로 대화의 유해성 여부를 판단하고 관련 대사를 출력하라. 유해성 대사가 없다면 없음 이라고 출력하라. " + \
                     "설명란에 유해성 판단의 근거 및 관련 줄거리를 총 125자 이내로 간결하게 요약하라."

            dict = {"instruction": prompt,
                    "input": subtitle.replace("['", '').replace("']", '').replace("'", ''),
                    "output": str(infer).replace("'", '"').replace('\"','')}
            data.append(dict)
        json.dump(data, make_file, ensure_ascii=False, indent=4)


few_shot_example = pd.read_excel('few-shot-example.xlsx')
few_shot_data = []
for input, 대사, 설명 in zip(few_shot_example['대사'], few_shot_example['유해성 있는 대사'], few_shot_example['설명']):
    dict = {"input":input.replace("\xa0", ""),
            "output": "유해성 대사: " + 대사 + " \n" + "설명: " + 설명}
    print(dict)
    few_shot_data.append(dict)

with open('few_shot.json', 'w', encoding='utf-8') as make_file:
    json.dump(few_shot_data, make_file, ensure_ascii=False, indent=4)