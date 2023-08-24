import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
from pytorch_lightning import LightningModule
from transformers import get_linear_schedule_with_warmup
from torchmetrics import MetricCollection
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report
)
import logging
import string
from collections import Counter

from dataset import LongKeT5Dataset


class KoreanLongT5Model(LightningModule):
    def __init__(self, cfg, trainer):
        super().__init__()
        self.cfg = cfg
        self.trainer = trainer
        self.config = AutoConfig.from_pretrained(cfg.model_name_or_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(cfg.model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path)
        self.tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
        self.max_length = cfg.max_length
        self.num_labels = cfg.num_labels
        self.configure_metrics()

        self.label_list = ['전체관람가', '12세관람가', '15세관람가', '18세관람가']
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
        outputs = self.model(input_ids = batch['input_ids'], attention_mask = batch['attention_mask'],
                             labels = batch['labels'])
        return outputs.loss

    def tokenized_decode(self, token_ids):
        pred_tokens = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        pred_tokens = self.normalize_text(pred_tokens)
        if pred_tokens in self.label_list:
            prediction = self.label_list.index(pred_tokens.strip())
            return prediction
        else:
            return 4

    def normalize_text(self, text):
        def white_space_fix(text):
            return text.replace(" ", "")

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)
        return white_space_fix(remove_punc(text))

    def configure_metrics(self, mode="val"):
        self.metric_collections = {f"{mode}_precision": 0,
                                   f"{mode}_recall": 0,
                                   f"{mode}_accuracy": 0,
                                   f"{mode}_f1": 0}

    def compute_metrics(self, mode="val"):
        if mode == 'val':
            accuracy = accuracy_score(self.val_output['labels'], self.val_output['preds'])
            precision = precision_score(self.val_output['labels'], self.val_output['preds'], average='macro')
            recall = recall_score(self.val_output['labels'], self.val_output['preds'], average='macro')
            f1 = f1_score(self.val_output['labels'], self.val_output['preds'], average='macro')
        elif mode =='test':
            accuracy = accuracy_score(self.test_output['labels'], self.test_output['preds'])
            precision = precision_score(self.test_output['labels'], self.test_output['preds'], average='macro')
            recall = recall_score(self.test_output['labels'], self.test_output['preds'], average='macro')
            f1 = f1_score(self.test_output['labels'], self.test_output['preds'], average='macro')
        self.metric_collections[f"{mode}_precision"] = precision
        self.metric_collections[f"{mode}_recall"] = recall
        self.metric_collections[f"{mode}_accuracy"] = accuracy
        self.metric_collections[f"{mode}_f1"] = f1

    def training_step(self, batch, batch_idx):
        loss = self(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self(batch)
        predictions = self.model.generate(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        labels = batch['labels']

        for pred, label in zip(predictions, labels):
            pred = self.tokenized_decode(pred)
            label = self.tokenized_decode(label)
            self.val_output['preds'].append(pred)
            self.val_output['labels'].append(label)

        logging.info(f"***EPOCH {self.current_epoch}, preds: {self.val_output['preds'][0]}, labels: {self.val_output['labels'][0]}")
        self.log('val_loss', loss, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True, logger=True)
        return loss

    def on_validation_epoch_end(self):
        self.compute_metrics(mode="val")
        self.log_dict(self.metric_collections, on_epoch=True, sync_dist=True, prog_bar=True, logger=True)
        self.configure_metrics()

    def test_step(self, batch, batch_idx):
        self.configure_metrics(mode="test")
        loss = self(batch)
        predictions = self.model.generate(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        labels = batch['labels']

        for pred, label in zip(predictions, labels):
            pred = self.tokenized_decode(pred)
            label = self.tokenized_decode(label)
            self.test_output['preds'].append(pred)
            self.test_output['labels'].append(label)

    def on_test_epoch_end(self):
        target_names = ['전체관람가', '12세관람가', '15세관람가', '18세관람가']
        logging.info(classification_report(self.test_output['labels'], self.test_output['labels'], target_names=target_names))

        self.compute_metrics(mode="test")
        self.log_dict(self.metric_collections, on_epoch=True, sync_dist=True, prog_bar=True, logger=True)

    def _loader(self, data_config, split):
        dataset = LongKeT5Dataset(data_dir=data_config.data_dir,
                                        tokenizer=self.tokenizer,
                                        max_length=self.max_length,
                                        split=split)
        return torch.utils.data.DataLoader(dataset,
                                           collate_fn=dataset.collate_fn,
                                           **data_config.dataloader_params)

    def train_dataloader(self):
        return self._loader(self.cfg.data, 'train')

    def val_dataloader(self):
        return self._loader(self.cfg.data, 'val')

    def test_dataloader(self):
        return self._loader(self.cfg.data, 'test')