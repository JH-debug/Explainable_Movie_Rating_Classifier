import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
import hydra
from omegaconf import DictConfig, OmegaConf
from model.kcelectra_model import KcElectradModel
from model.kobigbird_model import KoBigbirdModel
from model.longket5_model import KoreanLongT5Model
from model.longformer_roberta_model import RobertaLongformerModel
import logging


@hydra.main(config_path='config', config_name='KoBigbird')
def main(cfg: DictConfig):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    lr_monitor = pl.callbacks.LearningRateMonitor()
    early_stop = pl.callbacks.EarlyStopping(monitor='val_loss', patience=cfg.patience, mode='min')
    checkponiter = pl.callbacks.ModelCheckpoint(dirpath=cfg.checkpoint_dir,
                                                filename='model_{epoch:d}-{val_loss:.2f}',
                                                verbose=True, save_top_k=3, monitor='val_loss',
                                                mode='min', save_on_train_epoch_end=True, save_last=True
                                                )

    if len(cfg.num_gpus) > 1:
        ddp = DDPStrategy(process_group_backend='gloo', find_unused_parameters=True, static_graph=True)
        trainer = pl.Trainer(**cfg.trainer,
                             callbacks=[lr_monitor, checkponiter, early_stop],
                             strategy=ddp,
                             logger=WandbLogger(project=cfg.project, name=cfg.name)
                             )
    else:
        trainer = pl.Trainer(**cfg.trainer,
                             callbacks=[lr_monitor, checkponiter, early_stop],
                             logger=WandbLogger(project=cfg.project, name=cfg.name)
                             )
    if 'kcelectra' in cfg.model_name_or_path.lower():
        model = KcElectradModel(cfg, trainer)
    elif 'kobigbird' in cfg.model_name_or_path.lower():
        model = KoBigbirdModel(cfg, trainer)
    elif 'long-t5' in cfg.model_name_or_path.lower():
        model = KoreanLongT5Model(cfg, trainer)
    elif 'roberta' in cfg.model_name_or_path.lower():
        model = RobertaLongformerModel(cfg, trainer)

    trainer.fit(model,)
                # ckpt_path=cfg.test_model_path)


if __name__ == "__main__":
    main()