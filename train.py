from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.model_summary import summarize
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

import src.data_modules
import src.interfaces


def create_trainer(config):
    logger = None

    # WandB Logging
    if config.get("wandb") is not None:
        # Pass in wandb.init(config=) argument to get the nice 'x.y.0.z' hparams logged
        # Can pass in config_exclude_keys='wandb' to remove certain groups

        wandb_cfg = config.pop("wandb")
        logger = WandbLogger(
            config=OmegaConf.to_container(config, resolve=True),
            **wandb_cfg,
        )

    callbacks = [
        LearningRateMonitor(
            logging_interval="step",
        )
    ]
    # checkpointing
    if config.trainer.enable_checkpointing:
        callbacks.append(
            ModelCheckpoint(
                dirpath=Path(config.train.results_dir) / "checkpoints",
                filename=f"epoch={{epoch:02d}}-{config.train.monitor_metric}={{{config.train.monitor_metric}:.3f}}",
                save_top_k=3,  # how many best models to keep
                monitor=config.train.monitor_metric,  # metric to monitor
                mode="min",  # minimize or maximize the metric
            )
        )

    # Profiler
    profiler = None
    if config.trainer.get("profiler", None) is not None:
        profiler = hydra.utils.instantiate(config.trainer.profiler)
        config.trainer.pop("profiler")

    # Configure ddp automatically
    if config.trainer.accelerator == "gpu" and config.trainer.devices > 1:
        print("ddp automatically configured, more than 1 gpu used!")
        config.trainer.strategy = "ddp"

    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        profiler=profiler,
        **config.trainer,
    )
    return trainer


def train(config):
    if config.train.seed is not None:
        pl.seed_everything(config.train.seed, workers=True)
    config.train.results_dir = str(
        Path(config.train.results_dir) / wandb.run.project / f"{wandb.run.name}-{wandb.run.id}"
    )
    trainer = create_trainer(config)
    interface = getattr(src.interfaces, config.interface.pop("name"))(config)
    summary = summarize(interface, max_depth=2)
    print(summary)
    print(f"Total parameters: {summary.total_parameters}")
    print(f"Trainable parameters: {summary.trainable_parameters}")

    for name, p in interface.model.named_parameters():
        print(f"{name:55s} shape={tuple(p.shape)}  requires_grad={p.requires_grad}")

    data_module = getattr(src.data_modules, config.dataset.pop("name"))(**config.dataset)

    # Run initial validation epoch (useful for debugging, finetuning)
    if config.train.validate_at_start:
        print("Running validation before training")
        trainer.validate(interface, datamodule=data_module)

    if config.train.ckpt is not None:
        trainer.fit(interface, ckpt_path=config.train.ckpt, datamodule=data_module)
    else:
        trainer.fit(interface, datamodule=data_module)
    if config.train.test:
        trainer.test(interface, datamodule=data_module)


@hydra.main(config_path="configs", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    OmegaConf.set_struct(cfg, False)  # Allow writing keys

    # Track with wandb
    wandb_cfg = cfg["wandb"]
    wandb.init(**wandb_cfg, config=OmegaConf.to_container(cfg, resolve=True))

    train(cfg)


if __name__ == "__main__":
    main()
