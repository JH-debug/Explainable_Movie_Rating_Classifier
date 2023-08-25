import os
import torch
import transformers
from transformers import AutoTokenizer, GPTNeoXTokenizerFast, AutoModelForCausalLM, BitsAndBytesConfig
from peft import (get_peft_model, get_peft_model_state_dict, prepare_model_for_int8_training,
                  IA3Config, LoraConfig)
from datasets import load_dataset

from utils import Prompter


os.environ["WANDB_PROJECT"] = "유해성"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

model_id = "nlpai-lab/kullm-polyglot-5.8b-v2"
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_use_double_quant=True,
    bnb_8bit_quant_type="nf4",
    bnb_8bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
# tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config,
                                             # torch_dtype=torch.float16,
                                             device_map="auto")
                                             # {"":0})
model = prepare_model_for_int8_training(model)
# config = LoraConfig(
#     r=2,
#     lora_alpha=4,
#     target_modules=["query_key_value", "xxx"],
#     lora_dropout=0.05,
#     bias="none",
#     task_type="CAUSAL_LM",
# )
config=IA3Config(task_type="CAUSAL_LM")

model = get_peft_model(model, config)
model.resize_token_embeddings(len(tokenizer))
print_trainable_parameters(model)
# for i in model.named_parameters():
#     print(f"{i[0]} -> {i[1].device}")

data = load_dataset("json", data_files={"train": "data/llm_train.json", "test": "data/llm_test.json"})
prompter = Prompter()
cutoff_len = 2048
batch_size = 2
micro_batch_size = 1
gradient_accumulation_steps = 1
# gradient_accumulation_steps = batch_size // micro_batch_size

world_size = int(os.environ.get("WORLD_SIZE", 1))
ddp = world_size != 1
if ddp:
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    gradient_accumulation_steps = gradient_accumulation_steps // world_size

def tokenize(prompt, add_eos_token=True):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        return_tensors=None,
    )
    if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < 2048
            and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()

    return result

def generate_and_tokenize_prompt(data_point):
    full_prompt = prompter.generate_prompt(
        data_point["instruction"],
        data_point["input"],
        data_point["output"],
    )
    tokenized_full_prompt = tokenize(full_prompt)

    return tokenized_full_prompt

train_data = data["train"].map(generate_and_tokenize_prompt)
test_data = data["test"].map(generate_and_tokenize_prompt)

if not ddp and torch.cuda.device_count() > 1:
    # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
    model.is_parallelizable = True
    model.model_parallel = True

trainer = transformers.Trainer(
    model=model,
    train_dataset= train_data,
    eval_dataset=test_data,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=100,
        num_train_epochs=5,
        learning_rate=5e-4,
        fp16=True,
        logging_steps=1,
        optim='adamw_torch',
        evaluation_strategy="epoch",
        save_strategy="epoch",
        output_dir="outputs",
        save_total_limit=3,
        load_best_model_at_end=True,
        report_to="wandb",
        run_name="llm_kullum",
        ddp_find_unused_parameters=False if ddp else None),
    data_collator=transformers.DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    ),
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
old_state_dict = model.state_dict
model.state_dict = (lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())).__get__(
    model, type(model)
)

trainer.train(resume_from_checkpoint=None)

# save the model
output_dir = "outputs"
trainer.save_pretrained(output_dir)