import json
from typing import Union

class Prompter(object):
    __slots__ = ("template", "response", "_verbose")

    def __init__(self, verbose: bool = False):
        self.template = "명령어:\n{instruction}\n\n### 입력:\n{input}\n\n### 응답:\n"
        self.response = "### 응답:"
        self._verbose = verbose

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:

        if input:
            res = self.template.format(
                instruction=instruction, input=input
            )

        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.response)[-1].strip()



if __name__ == "__main__":
    from datasets import load_dataset
    data = load_dataset("json", data_files={"train": "data/llm_train.json", "test": "data/llm_test.json"})
    print(data['train'])
    prompter = Prompter()
    for i in range(2):
        print(prompter.generate_prompt(data["train"][i]["instruction"], data["train"][i]["input"], data["train"][i]["output"]))

    from train import generate_and_tokenize_prompt
    train_data = data["train"].map(generate_and_tokenize_prompt)
    print(train_data[0])