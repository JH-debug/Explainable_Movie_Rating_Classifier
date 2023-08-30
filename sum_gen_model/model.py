import torch
from transformers import (AutoConfig, AutoModel, AutoTokenizer,
                          AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq,
                          AutoModelForCausalLM)
from pytorch_lightning import LightningModule
from transformers import get_linear_schedule_with_warmup
import logging
from torchmetrics.functional.text.bert import bert_score
from torchmetrics.text import BLEUScore, SacreBLEUScore
from torchmetrics.text.rouge import ROUGEScore
import numpy as np
import json

from sum_gen_dataset import SumGenDataset


class SumGenModel(LightningModule):
    def __init__(self, cfg, trainer):
        super().__init__()
        self.cfg = cfg
        self.trainer = trainer
        self.config = AutoConfig.from_pretrained(cfg.model_name_or_path)
        if 'gpt' in cfg.model_name_or_path:
            self.model = AutoModelForCausalLM.from_pretrained(cfg.model_name_or_path)
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path)
        self.tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model,
                                                        label_pad_token_id=self.tokenizer.pad_token_id)
        self.max_length = cfg.max_length
        self.configure_metrics()

        self.val_output = {'preds': [], 'labels': []}
        self.test_output = {'preds': [], 'labels': []}

    def setup(self, stage):
        if stage == 'fit' or stage is None:
            total_devices = self.trainer.num_devices * self.trainer.num_nodes
            train_batches = len(self.train_dataloader()) // total_devices
            self.train_steps = (self.cfg.trainer.max_epochs * train_batches) // self.trainer.accumulate_grad_batches

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.cfg.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.cfg.lr, eps=self.cfg.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.cfg.warmup_steps, num_training_steps=self.train_steps
        )
        return [optimizer], [scheduler]

    def forward(self, batch):
        if 'gpt' in self.cfg.model_name_or_path:
            outputs = self.model(input_ids=batch['input_ids'], labels=batch['labels'])
        else:
            outputs = self.model(**batch)
        return outputs.loss
    def configure_metrics(self):
        self.bleu_score = BLEUScore(n_gram=1, smooth=True)
        self.sacre_bleu = SacreBLEUScore(n_gram=1, smooth=True)
        self.rouge_score = ROUGEScore(tokenizer=AutoTokenizer.from_pretrained("klue/bert-base"))

    def training_step(self, batch, batch_idx):
        loss = self(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self(batch)
        predictions = self.model.generate(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        if 'gpt' in self.cfg.model_name_or_path:
            labels = batch['labels']
        else:
            labels = batch['decoder_input_ids']

        for pred, label in zip(predictions, labels):
            pred = self.tokenizer.decode(pred, skip_special_tokens=True)
            label = self.tokenizer.decode(label, skip_special_tokens=True)
            self.rouge_score.update(pred, label)
            self.bleu_score.update(pred, label)
            self.sacre_bleu.update(pred, label)
            self.val_output['preds'].append(pred)
            self.val_output['labels'].append(label)

        logging.info(f"***EPOCH {self.current_epoch}, preds: {self.val_output['preds'][0]}, labels: {self.val_output['labels'][0]}")
        self.log('val_loss', loss, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True, logger=True)
        return loss

    def on_validation_epoch_end(self):
        bert_scores = bert_score(self.val_output['preds'], self.val_output['labels'],
                                 model_name_or_path="klue/bert-base", verbose=True)

        for key in bert_scores.keys():
            self.log_dict({key: np.nanmean(bert_scores[key])}, on_epoch=True, sync_dist=True, prog_bar=True, logger=True)
        self.log_dict(self.rouge_score.compute(), on_epoch=True, sync_dist=True, prog_bar=True, logger=True)
        self.log('bleu_score', self.bleu_score.compute(), on_epoch=True, sync_dist=True, prog_bar=True, logger=True)
        self.log('sacre_bleu', self.sacre_bleu.compute(), on_epoch=True, sync_dist=True, prog_bar=True, logger=True)
        self.configure_metrics()

    def test_step(self, batch, batch_idx):
        loss = self(batch)
        predictions = self.model.generate(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        if 'gpt' in self.cfg.model_name_or_path:
            labels = batch['labels']
        else:
            labels = batch['decoder_input_ids']

        for pred, label in zip(predictions, labels):
            pred = self.tokenizer.decode(pred, skip_special_tokens=True)
            label = self.tokenizer.decode(label, skip_special_tokens=True)
            self.rouge_score.update(pred, label)
            self.bleu_score.update(pred, label)
            self.sacre_bleu.update(pred, label)
            self.test_output['preds'].append(pred)
            self.test_output['labels'].append(label)

    def on_test_epoch_end(self):
        bert_scores = bert_score(self.test_output['preds'], self.test_output['labels'],
                                 model_name_or_path="klue/bert-base", verbose=True)
        for key in bert_scores.keys():
            self.log_dict({key: np.nanmean(bert_scores[key])}, on_epoch=True, sync_dist=True, prog_bar=True, logger=True)
        self.log_dict(self.rouge_score.compute(), on_epoch=True, sync_dist=True, prog_bar=True, logger=True)
        self.log('bleu_score', self.bleu_score.compute(), on_epoch=True, sync_dist=True, prog_bar=True, logger=True)
        self.log('sacre_bleu', self.sacre_bleu.compute(), on_epoch=True, sync_dist=True, prog_bar=True, logger=True)
        self.configure_metrics()

        with open(f'{self.cfg.test_save_path}', 'w', encoding='utf-8') as f:
            json.dump(self.test_output, f, ensure_ascii=False, indent=4)

    def _loader(self, data_config, split):
        dataset = SumGenDataset(tokenizer=self.tokenizer,
                                  max_length=self.max_length,
                                  split=split)
        return torch.utils.data.DataLoader(dataset,
                                           collate_fn=self.data_collator,
                                           **data_config.dataloader_params)

    def train_dataloader(self):
        return self._loader(self.cfg.data, 'train')

    def val_dataloader(self):
        return self._loader(self.cfg.data, 'test')

    def test_dataloader(self):
        return self._loader(self.cfg.data, 'test')