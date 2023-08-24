import os
import pandas as pd
import torch
from langchain.llms import OpenAI
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.prompts import StringPromptTemplate
from pydantic import BaseModel, validator
from langchain import PromptTemplate, HuggingFaceHub, LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
from tqdm import tqdm


os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.environ['OPENAI_API_KEY'] = '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'

# """\n"유해성관련기준": [{"주제": "해당 연령층의 정서와 가치관, 인격형성 등에 끼칠 영향 또는 그 이해와 수용정도"},""" + \
# """{"선정성": "신체 노출과 애무, 정사장면 등 성적 행위의 표현 정도"}, {"폭력성": "신체 분위, 도구 등을 이용한 물리적 폭력과 성폭력, 이로 인해 발생한 상해, 유혈, 신체훼손, 고통 등의 빈도와 표현정도"},""" + \
# """{"대사": "욕설, 비속어, 저속어 등의 빈도와 표현 정도"},""" + \
# """{"공포": "긴장감과 불안감, 그 자극과 위협으로 인한 정신적 충격 유발정도"},""" + \
# """{"약물": "소재나 수단으로 다루어진 약물 등의 표현 정도"},""" + \
# """{"모방위험": "살인, 마약, 자살, 학교 폭력, 따돌림, 청소년 비행과 무기류 사용, 범죄기술 등에 대한 모방심리를 고무, 자극하는 정도"}]}""" + \

class MovieExplainerPromptTemplate(StringPromptTemplate, BaseModel):
    """Prompt template for the movie explainer task."""
    # @validator("input_variables")
    # def validate_input_variables(cls, **v):
    #     """Validate that the input variables are correct."""
    #     for var in v:
    #         if var not in ["len", "idx", "subtitles"]:
    #             raise ValueError("input_variables must be len, idx, subtitles.")
    #     if len(v) != 1 or "subtitles" not in v:
    #         raise ValueError("subtitles must be the only input_variable.")
    #     return v

    def format(self, **kwargs) -> str:
        # Generate the prompt to be sent to the language model
        prompt = """"대화": [""" + kwargs["subtitles"] + "]" +\
        """\n"유해성기준": [{"욕설, 비속어, 저속어": "저속한 욕설, 비속어, 저속어 표현"},""" +\
        """{"언어 습관이 미치는 영향": "공격적이고 수치심을 느끼게 하는 거친 표현이 있으나 내용 전개상 수용가능한 표현"},""" +\
        """{"차별적 인권침해적 언어 사용의 표현": "자극적이고 혐오스러운 성적 표현이나 정서적 인격적인 모욕감이나 수치심을 유발하는 표현"}]}""" +\
        "\n\n위의 대화는 영화의 유해성을 파악하기 위해 영화 대사를 " + kwargs["total_len"] + "개로 나누어 세그먼트로 분류한 것 " + kwargs["total_len"] + " 번째 중 " + kwargs["idx"] + "번째이다. "+\
        "대화를 읽고 유해성기준을 기반으로 세그먼트의 유해성 여부를 판단하고 관련 대사를 출력하라. 유해성 대사가 없다면 '없음'이라고 출력하라. " +\
        "설명란에 유해성 판단의 근거 및 관련 줄거리를 총 125자 이내로 간결하게 요약하라." +\
        "\n\n출력" +\
        "\n유해성 대사: " +\
        "\n설명:"
        return prompt

    def _prompt_type(self):
        return "movie-explainer"

prompt = MovieExplainerPromptTemplate(input_variables=["total_len", "idx", "subtitles"])
llm = OpenAI(model_name = "gpt-3.5-turbo", max_tokens=256)
llm_chain = LLMChain(prompt=prompt, llm=llm)

# loader = CSVLoader(file_path="../data/kobigbird_data.csv", source_column="Movie_name")
# data = loader.load()
data = pd.read_csv("../data/df_concat.csv", encoding='utf-8')
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 512,
                                               chunk_overlap=10,
                                               # add_start_index=True,
                                               separators=[" ", ",", "\n"])


for movie_name in tqdm((data['Movie_name'])):
    if movie_name not in ('118_보스베이비', '100_트롤헌터 라이즈 오브 타이탄', '기생충', '신세계'):
        age = data[data['Movie_name'] == movie_name]['age'].values[0]
        print("영화 이름 : " + movie_name + " / 연령 : " + age)
        one_movie_data = data[data['Movie_name'] == movie_name]['Subtitle'].values[0]
        split_data = text_splitter.create_documents([one_movie_data])
        len_split = len(split_data)
        # print(len_split)

        subtitle_list = []
        prompt_list = []
        prediction_list = []

        for i in range(len_split):
            data_split = split_data[i].page_content
            subtitles = str(data_split)
            index = str(i)
            # print("=====================================================")
            # print(str(i) + "번째 대사")
            # print("Prompt: ")
            # print(prompt.format(total_len=str(len_split), idx=index, subtitles=subtitles))
            #
            # print("\nPrediction: ")
            # print(llm_chain.run(total_len=str(len_split), idx=index, subtitles=subtitles))

            subtitle_list.append(subtitles)
            prompt_list.append(prompt.format(total_len=str(len_split), idx=index, subtitles=subtitles))
            prediction_list.append(llm_chain.run(total_len=str(len_split), idx=index, subtitles=subtitles))

        infer_data = pd.DataFrame({'subtitle': subtitle_list, 'prompt': prompt_list, 'prediction': prediction_list})
        infer_data.to_csv(f'../data/LLM_infer/{age}/{movie_name}.csv', index=False, encoding='utf-8-sig')