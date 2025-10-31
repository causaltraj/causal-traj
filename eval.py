import argparse

import pytorch_lightning as pl
from omegaconf import OmegaConf

import src.data_modules
import src.interfaces


def run_test(config):
    if config.train.seed is not None:
        pl.seed_everything(config.train.seed, workers=True)

    trainer = pl.Trainer(
        **config.trainer,
    )

    interface = getattr(src.interfaces, config.interface.pop("name"))(config)
    data_module = getattr(src.data_modules, config.dataset.pop("name"))(**config.dataset)

    trainer.test(interface, datamodule=data_module, ckpt_path=config.train.ckpt)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-config-path", type=str, required=True)
    p.add_argument("--test-config-path", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--test-size", default=None, type=int)
    p.add_argument("--bsz", default=1024, type=int)
    p.add_argument("--mean-sampling", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.base_config_path)
    test_cfg = OmegaConf.load(args.test_config_path)
    cfg = OmegaConf.merge(cfg, test_cfg)
    OmegaConf.set_struct(cfg, False)
    cfg.train.ckpt = args.ckpt
    cfg.dataset.val_size = args.test_size
    cfg.dataset.val_bsz = args.bsz
    cfg.interface.mean_sampling = args.mean_sampling
    run_test(cfg)


if __name__ == "__main__":
    main()
