import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
import hydra
from omegaconf import DictConfig, OmegaConf
from model.kobigbird_model import KoBigbirdModel
from model.kcelectra_model import KcElectradModel
from model.longket5_model import KoreanLongT5Model
import logging


@hydra.main(config_path='config', config_name='KoBigbird')
def main(cfg: DictConfig):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    trainer = pl.Trainer(**cfg.trainer)
    if 'kcelectra' in cfg.model_name_or_path.lower():
        model = KcElectradModel(cfg, trainer)
    elif 'kobigbird' in cfg.model_name_or_path.lower():
        model = KoBigbirdModel(cfg, trainer)
    elif 't5-base' in cfg.model_name_or_path.lower():
        model = KoreanLongT5Model(cfg, trainer)

    trainer.test(model=model, ckpt_path=cfg.test_model_path)


if __name__ == "__main__":
    main()