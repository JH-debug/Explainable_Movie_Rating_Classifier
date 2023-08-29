from torch.utils.data import Dataset
import json

class SumGenDataset(Dataset):
    def __init__(self, tokenizer, max_length=4096, split='train'):
        self.data = json.load(open(f"/home/jhlee/Explainable_Movie_Rating_Classifier/LLM/data/llm_{split}.json", "r", encoding="utf-8"))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        instruction = self.data[idx]["instruction"]
        input = self.data[idx]["input"]
        output = self.data[idx]["output"]

        input = instruction + input

        inputs = self.tokenizer.encode_plus(
            input,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        labels = self.tokenizer.encode_plus(
            output,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        ).input_ids


        return {'input_ids': inputs['input_ids'].squeeze(0),
                'attention_mask': inputs['attention_mask'].squeeze(0),
                'labels': labels.squeeze(0)}

    # def collate_fn(self, batch):
    #     batch = self.tokenizer.pad(batch, return_tensors='pt')
    #     return batch


if __name__ == "__main__":
    import torch
    from transformers import AutoTokenizer, DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM

    tokenizer = AutoTokenizer.from_pretrained("KETI-AIR-Downstream/long-ke-t5-base-summarization")
    dataset = SumGenDataset(tokenizer=tokenizer, max_length=2048, split='train')
    print(dataset[0])

    model = AutoModelForSeq2SeqLM.from_pretrained("KETI-AIR-Downstream/long-ke-t5-base-summarization")
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=tokenizer.pad_token_id)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, collate_fn=data_collator)
    print(next(iter(dataloader)))