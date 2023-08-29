import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftConfig, PeftModel
import numpy as np
from datasets import load_dataset
import pandas as pd
import json
from tqdm import tqdm
from random import sample
import re

from utils import Prompter

# tokenizer = AutoTokenizer.from_pretrained("EleutherAI/polyglot-ko-5.8b")
tokenizer = AutoTokenizer.from_pretrained("nlpai-lab/kullm-polyglot-5.8b-v2")
config = PeftConfig.from_pretrained("outputs/checkpoint-92415")
cls_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path,
                                                 torch_dtype=torch.float16,
                                                 low_cpu_mem_usage=True).cuda()
cls_model = PeftModel.from_pretrained(cls_model, "outputs/checkpoint-92415")

cls_model.eval()
cls_model.config.max_length = 2048
cls_model.config.pad_token_id = 0
# cls_model.config.max_position_embeddings = 512
print("모델 최대 토큰 길이: ", cls_model.config.max_position_embeddings)

data = load_dataset("json", data_files={"train": "data/llm_train.json", "test": "data/llm_test.json"})
test_data = data["test"]

sent_lens = [len(tokenizer(s).input_ids) for s in tqdm(test_data["input"])]
print('Few shot 케이스 토큰 평균 길이: ', np.mean(sent_lens))
print('Few shot 케이스 토큰 최대 길이: ', np.max(sent_lens))
print('Few shot 케이스 토큰 길이 표준편차: ',np.std(sent_lens))
print('Few shot 케이스 토큰 길이 80%: ',np.percentile(sent_lens, 80))

few_shot_data = json.load(open(f"data/few_shot.json", "r", encoding="utf-8"))

def build_prompt_text(input):
    # instruction = test_data['instruction'][0]
    few_shot_prompt = ''
    instruction = "유해성기준을 기반으로 대화의 유해성 여부를 판단하고 관련 대사를 출력하라. "

    few_shot_examples = sample(few_shot_data, 4)
    for few_shot_example in few_shot_examples:
        # few_shot_prompt += few_shot_example['input'] + "\n 정답:\n" + few_shot_example['output'] + "<|endoftext|>\n\n"
        few_shot_prompt += "\n\n명령어:\n" + instruction + "\n### 입력:\n" + few_shot_example['input'] + "\n### 응답:\n" + few_shot_example['output']
        tokens = tokenizer(few_shot_prompt).input_ids
        if len(tokens) > 1800:
            few_shot_prompt = few_shot_prompt[:1800]
            break

    return f"{few_shot_prompt}\n\n명령어:\n{instruction}\n### 입력:\n{input}\n### 응답:\n"
    # return f"{instruction}\n\n{few_shot_prompt}\n{input}\n 정답: "

def clean_text(sent):
    sent_clean = re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣ\\s]", "", sent)
    return sent_clean

prompter = Prompter()
pipe = pipeline("text-generation", model=cls_model, tokenizer=tokenizer, device=0)

with torch.no_grad():
    real_labels = []
    pred_tokens = []

    total_len = len(test_data[15:20])

    for i, (test_sent, test_label) in tqdm(enumerate(zip(test_data['input'][15:20], test_data['output'][15:20])), total=total_len):
        cleaned_sent = clean_text(test_sent)
        prompt_text = build_prompt_text(cleaned_sent)

        pred = pipe(prompt_text, max_length=1024, do_sample=True,
                    pad_token_id=0,
                    eos_token_id=2)[0]['generated_text']
        # tokens = tokenizer(prompt_text, return_tensors="pt")
        # token_ids, attn_mask = tokens.input_ids.cuda(), tokens.attention_mask.cuda()
        # gen_tokens = cls_model.generate(input_ids=token_ids, attention_mask=attn_mask, do_sample=True,
        #                                 max_new_tokens=512,
        #                                 eos_token_id=2,
        #                                 pad_token_id=0)

        # pred = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        print(pred)
        pred = prompter.get_response(pred)
        print('\n', pred)

        pred_tokens.append(pred)
        real_labels.append(test_label)