import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from pytorch_lightning import LightningModule
from transformers import get_linear_schedule_with_warmup
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score, MulticlassConfusionMatrix
from sklearn.metrics import classification_report
import logging

from dataset import KoBigbirdMovieDataset
from model.longformer_model import RobertaLongForSequenceClassification, make_RobertaLongForSequenceClassification


class RobertaLongformerModel(LightningModule):
    def __init__(self, cfg, trainer):
        super().__init__()
        self.cfg = cfg
        self.trainer = trainer
        self.config = AutoConfig.from_pretrained(cfg.model_name_or_path, num_labels=cfg.num_labels)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path)
        self.tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
        if cfg.resume_training or cfg.test:
            self.model = RobertaLongForSequenceClassification.from_pretrained(cfg.model_name_or_path,
                                                                              config=self.config)
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(cfg.model_name_or_path,
                                                                            config=self.config)
            self.model, self.tokenizer = make_RobertaLongForSequenceClassification(self.model, self.tokenizer)
            print(self.model)
        self.max_length = cfg.max_length
        self.num_labels = cfg.num_labels
        self.configure_metrics()

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

    def configure_metrics(self):
        self.acc = MulticlassAccuracy(num_classes=self.num_labels)
        self.prec = MulticlassPrecision(num_classes=self.num_labels)
        self.recall = MulticlassRecall(num_classes=self.num_labels)
        self.f1 = MulticlassF1Score( num_classes=self.num_labels)
        self.metric_collections = MetricCollection({"precision": self.prec,
                                                    "recall": self.recall,
                                                    "accuracy": self.acc,
                                                    "f1": self.f1})
        self.confusion_matrix = MulticlassConfusionMatrix(num_classes=self.num_labels)

    def update_metrics(self, predictions, labels):
        predictions = predictions[labels != -100]
        labels = labels[labels != -100]
        return self.metric_collections.update(predictions, labels)

    def compute_metrics(self, mode="val"):
        return {f"{mode}_{k}": metric.compute() for k, metric in self.metric_collections.items()}

    def forward(self, batch):
        outputs = self.model(batch['input_ids'], batch['attention_mask'], labels = batch['labels'])
        return outputs.loss, outputs.logits

    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = outputs[0]
        self.log('train_loss', loss, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        loss, logits = outputs[:2]
        predictions = torch.argmax(logits, dim=-1)
        labels = batch['labels']
        outputs = self.update_metrics(predictions, labels)
        self.log('val_loss', loss, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True, logger=True)
        return loss

    def on_validation_epoch_end(self):
        metric_dict = self.compute_metrics(mode="val")
        self.log_dict(metric_dict, on_epoch=True, sync_dist=True, prog_bar=True, logger=True)
        self.metric_collections.reset()

    def test_step(self, batch, batch_idx):
        outputs = self(batch)
        loss, logits = outputs[:2]
        predictions = torch.argmax(logits, dim=-1)
        labels = batch['labels']
        self.test_output['preds'].append(predictions)
        self.test_output['labels'].append(labels)
        outputs = self.update_metrics(predictions,labels)

    def on_test_epoch_end(self):
        preds = torch.cat(self.test_output['preds'], dim=0)
        labels = torch.cat(self.test_output['labels'], dim=0)
        target_names = ['전체관람가', '12세관람가', '15세관람가', '18세관람가']
        logging.info(classification_report(labels.cpu().numpy(), preds.cpu().numpy(), target_names=target_names))

        metric_dict = self.compute_metrics(mode="test")
        self.log_dict(metric_dict, on_epoch=True, sync_dist=True, prog_bar=True, logger=True)
        self.metric_collections.reset()

    def _loader(self, data_config, split, use):
        dataset = KoBigbirdMovieDataset(data_dir=data_config.data_dir,
                                        tokenizer=self.tokenizer,
                                        max_length=self.max_length,
                                        split=split,
                                        use=use)
        return torch.utils.data.DataLoader(dataset,
                                           collate_fn=dataset.collate_fn,
                                           **data_config.dataloader_params)

    def train_dataloader(self):
        return self._loader(self.cfg.data, split='train', use=self.cfg.data.use)

    def val_dataloader(self):
        return self._loader(self.cfg.data, split='val', use=self.cfg.data.use)

    def test_dataloader(self):
        return self._loader(self.cfg.data, split='test', use=self.cfg.data.use)

