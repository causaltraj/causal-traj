import os
from copy import deepcopy
from mimetypes import init
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import wandb
from einops import rearrange
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

import src.models as models
from src.utils.data import cast_floats_by_trainer_precision, normalize, unnormalize
from src.utils.drawing import create_frames_from_trajectory, create_video_from_frames
from src.utils.prob import categorical_entropy, nll_mog_block2d, sample_mog_block2d


class BasePredictionInterface(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config, logger=True)

    def make_model_inputs_and_targets(self, batch: torch.tensor):
        raise NotImplementedError("Subclass must implement this method")

    def forward(self, x: torch.tensor):
        raise NotImplementedError("Subclass must implement this method")

    def training_step(self, batch, batch_idx):
        raise NotImplementedError("Subclass must implement this method")

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError("Subclass must implement this method")

    def test_step(self, batch, batch_idx):
        raise NotImplementedError("Subclass must implement this method")

    @property
    def learning_rate(self):
        return self.hparams.optimizer.lr

    @learning_rate.setter
    def learning_rate(self, value):
        self.hparams.optimizer.lr = value

    def configure_optimizers(
        self,
    ):
        lr = self.hparams.optimizer.lr
        weight_decay = self.hparams.optimizer.weight_decay

        # All parameters in the model
        all_parameters = list(self.model.parameters())

        # General parameters don't contain the special _optim key
        params = [p for p in all_parameters if not hasattr(p, "_optim")]

        # Create an optimizer with the general parameters
        optimizer = AdamW(
            params,
            lr=lr,
            weight_decay=weight_decay,
            betas=(self.hparams.get("beta1", 0.9), self.hparams.get("beta2", 0.95)),
        )

        # Add parameters with special hyperparameters
        hps = [getattr(p, "_optim") for p in all_parameters if hasattr(p, "_optim")]
        hps = [
            dict(s) for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))
        ]  # Unique dicts
        for hp in hps:
            params = [p for p in all_parameters if getattr(p, "_optim", None) == hp]
            optimizer.add_param_group({"params": params, **hp})

        # Print optimizer info
        keys = sorted(set([k for hp in hps for k in hp.keys()]))
        for i, g in enumerate(optimizer.param_groups):
            group_hps = {k: g.get(k, None) for k in keys}
            print(
                " | ".join(
                    [
                        f"Optimizer group {i}",
                        f"{len(g['params'])} tensors",
                    ]
                    + [f"{k} {v}" for k, v in group_hps.items()]
                )
            )
        # Create a lr scheduler
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=0.2)
        if self.hparams.optimizer.lr_schedule:
            total_steps = self.hparams.optimizer.lr_schedule.total_steps or getattr(
                self.trainer, "estimated_stepping_batches", None
            )
            if total_steps is None:
                raise ValueError(
                    "total_steps not set. Pass total_steps=... to the module "
                    "or let Lightning set trainer.estimated_stepping_batches by calling trainer.fit first."
                )
            max_lrs = [g.get("lr", lr) for g in optimizer.param_groups]
            scheduler = OneCycleLR(
                optimizer,
                max_lr=max_lrs,
                total_steps=total_steps,
                pct_start=self.hparams.optimizer.lr_schedule.pct_start,
                anneal_strategy="cos",
                cycle_momentum=False,
                div_factor=self.hparams.optimizer.lr_schedule.div_factor,
                final_div_factor=self.hparams.optimizer.lr_schedule.final_div_factor,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",  # OneCycleLR updates every step
                    "frequency": 1,
                },
            }
        else:
            return optimizer


class CausalScenarioPredictionInterface(BasePredictionInterface):
    def __init__(self, config):
        super().__init__(config)

        self.register_buffer("data_mean", torch.tensor(self.hparams.interface.data_mean))
        self.register_buffer("data_std", torch.tensor(self.hparams.interface.data_std))
        self.diff_as_target = self.hparams.interface.diff_as_target
        if self.hparams.interface.diff_as_target:
            self.register_buffer("diff_mean", torch.tensor(self.hparams.interface.diff_mean))
            self.register_buffer("diff_std", torch.tensor(self.hparams.interface.diff_std))
        self.model = getattr(models, self.hparams.model.name)(**self.hparams.model.args)
        self.validation_kidxs = torch.zeros(
            self.hparams.model.args.num_scenes,
            dtype=torch.long,
        )

    def make_model_inputs(self, batch: torch.tensor):

        batch_n = normalize(
            batch,
            self.data_mean,
            self.data_std,
        )
        if self.hparams.interface.diff_in_input:
            diff = torch.diff(batch, dim=1)
            diff_n = normalize(diff, self.diff_mean, self.diff_std)
            diff_n = torch.cat([torch.zeros_like(diff_n[:, :1]), diff_n], dim=1)
            input_n = torch.cat([batch_n, diff_n], dim=-1)
        else:
            input_n = batch_n
        return cast_floats_by_trainer_precision(input_n, precision=self.trainer.precision)

    def make_model_inputs_and_targets(self, batch: torch.tensor):
        if self.hparams.interface.diff_in_input or self.hparams.interface.diff_as_target:
            diff = torch.diff(batch, dim=1)
            diff_n = normalize(diff, self.diff_mean, self.diff_std)
            diff_n = torch.cat([torch.zeros_like(diff_n[:, :1]), diff_n], dim=1)

        batch_n = normalize(
            batch,
            self.data_mean,
            self.data_std,
        )
        if self.hparams.interface.diff_in_input:
            input_n = torch.cat([batch_n, diff_n], dim=-1)
        else:
            input_n = batch_n
        x = input_n[:, :-1]
        x = cast_floats_by_trainer_precision(x, precision=self.trainer.precision)

        if self.diff_as_target:
            y = diff_n[:, 1:]
            y = cast_floats_by_trainer_precision(y, precision=self.trainer.precision)
        else:
            y = batch_n[:, 1:]
            y = cast_floats_by_trainer_precision(y, precision=self.trainer.precision)

        return x, y, batch

    def forward(self, x: torch.tensor):
        pred, scene_logits, shrink, _ = self.model(x)
        return pred, scene_logits, shrink

    def training_step(self, batch, batch_idx):
        x, y, _ = self.make_model_inputs_and_targets(batch)
        pred, scene_logits, cov = self.forward(x)  # [b, t, k, a, 2], [b, t, k], [b, t, k, a, 2]
        nll = nll_mog_block2d(
            y=rearrange(y, "b t a d -> (b t) a d"),
            mu=rearrange(pred, "b t k a d -> (b t) k a d"),
            log_pi=rearrange(scene_logits, "b t k -> (b t) k"),
            L_packed=rearrange(cov, "b t k a d -> (b t) k a d"),
            eps=getattr(self.hparams.interface, "min_std", 5e-2),
            reduction="mean",
        )
        if pred.shape[2] > 1:
            entropy = categorical_entropy(scene_logits)
        else:
            entropy = torch.tensor(0.0).to(scene_logits.device)
        loss = nll - self.hparams.interface.entropy_weight * entropy
        record_step = {
            "trainer_loss": loss.item(),
            "trainer_nll": nll.item(),
            "trainer_entropy": entropy.item(),
        }

        self.log_dict(
            record_step,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            add_dataloader_idx=False,
            # sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, gt_path_original_scale = self.make_model_inputs_and_targets(batch)
        pred, scene_logits, cov = self.forward(x)  # [b, t, k, a, 2], [b, t, k], [b, t, k, a, 2]
        nll = nll_mog_block2d(
            y=rearrange(y, "b t a d -> (b t) a d"),
            mu=rearrange(pred, "b t k a d -> (b t) k a d"),
            log_pi=rearrange(scene_logits, "b t k -> (b t) k"),
            L_packed=rearrange(cov, "b t k a d -> (b t) k a d"),
            eps=getattr(self.hparams.interface, "min_std", 5e-2),
            reduction="mean",
        )
        if pred.shape[2] > 1:
            entropy = categorical_entropy(scene_logits)
        else:
            entropy = torch.tensor(0.0).to(scene_logits.device)
        loss = nll - self.hparams.interface.entropy_weight * entropy
        record_step = {
            "validation_nll": nll.item(),
            "validation_entropy": entropy.item(),
            "validation_loss": loss.item(),
        }

        samples_original_scale, k_idxs = self.sample(
            x[:, : self.hparams.interface.validation_prefix_length],
            max_length=self.hparams.interface.validation_max_length,
            num_paths=self.hparams.interface.validation_num_paths,
        )  # [b, num_paths, t, num_agents, 2]

        self.validation_kidxs += torch.bincount(
            k_idxs.flatten(), minlength=self.hparams.model.args.num_scenes
        ).cpu()
        metric_dict = self.compute_metrics(
            samples_original_scale,
            gt_path_original_scale[
                :,
                self.hparams.interface.validation_prefix_length : self.hparams.interface.validation_prefix_length
                + self.hparams.interface.validation_max_length,
            ],
        )

        record_step.update(metric_dict)

        self.log_dict(
            record_step,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            add_dataloader_idx=False,
        )

        if (
            self.trainer.global_step > 0
            and (self.trainer.current_epoch + 1) % self.hparams.interface.upload_every_n_epochs == 0
            and batch_idx == 0
            and self.hparams.interface.num_id_to_upload > 0
        ):
            print("sampling at validation step")
            sample_prefixes_original_scale = unnormalize(
                x[
                    : self.hparams.interface.num_id_to_upload,
                    self.hparams.interface.validation_prefix_length
                    - 10 : self.hparams.interface.validation_prefix_length,
                    :,
                    :2,  # only keep coordinates
                ],
                self.data_mean,
                self.data_std,
            )  # [b, t, num_agents, 2]
            samples_prefixes_original_scale = sample_prefixes_original_scale.unsqueeze(1).repeat(
                1, self.hparams.interface.num_paths_to_upload, 1, 1, 1
            )
            samples_original_scale = samples_original_scale[
                : self.hparams.interface.num_id_to_upload,
                : self.hparams.interface.num_paths_to_upload,
            ]
            samples_to_upload = (
                torch.cat([samples_prefixes_original_scale, samples_original_scale], dim=2)
                .cpu()
                .numpy()
            )  # [b, num_paths, t, num_agents, 2]

            video_dir = f"{os.path.expanduser(self.hparams.train.results_dir)}/samples"
            Path(video_dir).mkdir(parents=True, exist_ok=True)
            videos = []
            for i in range(samples_to_upload.shape[0]):
                for j in range(samples_to_upload.shape[1]):
                    frames = create_frames_from_trajectory(
                        samples_to_upload[i, j], game=self.hparams.interface.game
                    )
                    video_path = f"{video_dir}/sample_{i}_{j}.mp4"
                    create_video_from_frames(frames, video_path, fps=5)
                    videos.append(wandb.Video(video_path, format="mp4"))
            wandb.log({"sample": videos}, commit=False)

        return loss

    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()
        print(f"validation kidxs: {self.validation_kidxs}")
        self.validation_kidxs.zero_()

    def on_test_epoch_end(self):
        super().on_test_epoch_end()
        print(f"test kidxs: {self.validation_kidxs}")
        self.validation_kidxs.zero_()

    def sample(self, x: torch.tensor, max_length: int, num_paths: int, temperature: float = 1.0):
        """
        x: [b, t, num_agents, 2]
        """
        num_agents = x.shape[2]
        assert max_length > 1, "max_length must be greater than 1"
        samples = []
        init_reg_out, init_cls_out, init_cov, init_inference_cache = self.model(
            x, return_cache=True
        )
        init_cls_out = init_cls_out.detach()[:, -1]  # (b, k)
        init_reg_out = init_reg_out.detach()[:, -1:]  # [b, 1, k, a, 2]
        init_cov = init_cov.detach()[:, -1:]  # [b, 1, k, a, 2]

        samples = []
        k_idxs = []
        init_prev_output = unnormalize(x[:, -1:, :, :2], self.data_mean, self.data_std)
        for _ in range(num_paths):
            x, k_idx = sample_mog_block2d(
                rearrange(init_reg_out, "b t k a d -> (b t) k a d"),
                init_cls_out,
                rearrange(init_cov, "b t k a d -> (b t) k a d"),
                S=1,
                eps=getattr(self.hparams.interface, "min_std", 5e-2),
                mean_sampling=False,
                pi_temperature=temperature,
            )
            if self.hparams.interface.diff_as_target:
                output = unnormalize(x, self.diff_mean, self.diff_std) + init_prev_output
            else:
                output = unnormalize(x, self.data_mean, self.data_std)

            if self.hparams.interface.diff_in_input:
                input = torch.cat(
                    [
                        normalize(output, self.data_mean, self.data_std),
                        normalize(output - init_prev_output, self.diff_mean, self.diff_std),
                    ],
                    dim=-1,
                )
            else:
                input = normalize(output, self.data_mean, self.data_std)

            inference_cache = deepcopy(init_inference_cache)
            preds = [output]
            k_idxs_one_path = [k_idx]
            prev_output = output

            for t in range(max_length - 1):
                reg_out, cls_out, cov = self.model.generate(input, inference_cache)
                x, k_idx = sample_mog_block2d(
                    rearrange(reg_out, "b t k a d -> (b t) k a d"),
                    rearrange(cls_out, "b 1 k -> b k"),
                    rearrange(cov, "b t k a d -> (b t) k a d"),
                    S=1,
                    eps=getattr(self.hparams.interface, "min_std", 5e-2),
                    mean_sampling=False,
                    pi_temperature=temperature,
                )

                if self.hparams.interface.diff_as_target:
                    output = unnormalize(x, self.diff_mean, self.diff_std) + prev_output
                else:
                    output = unnormalize(x, self.data_mean, self.data_std)

                if self.hparams.interface.diff_in_input:
                    input = torch.cat(
                        [
                            normalize(output, self.data_mean, self.data_std),
                            normalize(output - prev_output, self.diff_mean, self.diff_std),
                        ],
                        dim=-1,
                    )
                else:
                    input = normalize(output, self.data_mean, self.data_std)
                preds.append(output)
                k_idxs_one_path.append(k_idx)
                prev_output = output
            k_idxs.append(torch.cat(k_idxs_one_path, dim=0))
            samples.append(torch.cat(preds, dim=1))  # [b, t, a, 2]
        samples = torch.stack(samples, dim=1)  # [b, num_paths, t, a, 2]
        k_idxs = torch.stack(k_idxs, dim=0)  # [b, num_paths, t]
        return samples, k_idxs

    def compute_metrics(self, samples, y):
        """
        samples: [b, num_paths, t, a, 2]
        y: [b, t, num_agents, 2]
        """
        output_dict = {}
        for i in range(0, samples.shape[2], 5):
            distances = (samples[:, :, : i + 5, :, :] - y[:, : i + 5, :, :].unsqueeze(1)).norm(
                p=2, dim=-1
            )  # [b, num_paths, t, num_agents]
            jade_path_agentwise = distances.mean(dim=-2)  # [b, num_paths, num_agents]
            jade_all = jade_path_agentwise.mean(dim=-1)  # [b, num_paths]
            jade_mean = jade_all.mean()
            jade_min = jade_all.min(dim=-1).values.mean()

            jfde_path_agentwise = distances[:, :, -1]  # [b, num_paths, num_agents]
            jfde_all = jfde_path_agentwise.mean(dim=-1)  # [b, num_paths]
            jfde_mean = jfde_all.mean()
            jfde_min = jfde_all.min(dim=-1).values.mean()

            ade_path_agentwise = distances.mean(dim=-2)  # [b, num_paths, num_agents]
            ade_agent_pathwise = rearrange(ade_path_agentwise, "b p a -> b a p")
            ade_min = ade_agent_pathwise.min(dim=-1).values.mean()

            fde_path_agentwise = distances[:, :, -1]  # [b, num_paths, num_agents]
            fde_agent_pathwise = rearrange(fde_path_agentwise, "b p a -> b a p")
            fde_min = fde_agent_pathwise.min(dim=-1).values.mean()

            output_dict[f"jade_mean_{i+5}frames"] = jade_mean
            output_dict[f"jade_min_{i+5}frames"] = jade_min
            output_dict[f"jfde_mean_{i+5}frames"] = jfde_mean
            output_dict[f"jfde_min_{i+5}frames"] = jfde_min
            output_dict[f"ade_min_{i+5}frames"] = ade_min
            output_dict[f"fde_min_{i+5}frames"] = fde_min

        return output_dict

    def test_step(self, batch, batch_idx):
        gt_path_original_scale = batch
        x = self.make_model_inputs(batch)
        record_step = {}

        samples_original_scale, k_idxs = self.sample(
            x[:, : self.hparams.test.prefix_length],
            max_length=self.hparams.test.max_length,
            num_paths=self.hparams.test.num_paths,
            temperature=self.hparams.test.temperature,
        )  # [b, num_paths, t, num_agents, 2]
        self.validation_kidxs += torch.bincount(
            k_idxs.flatten(), minlength=self.hparams.model.args.num_scenes
        ).cpu()

        metric_dict = self.compute_metrics(
            samples_original_scale,
            gt_path_original_scale[
                :,
                self.hparams.test.prefix_length : self.hparams.test.prefix_length
                + self.hparams.test.max_length,
            ],
        )
        record_step.update(metric_dict)

        if getattr(self.hparams.test, "feet2metre", False):
            for key, value in record_step.items():
                if "mean" in key:
                    record_step[key] = value * 28.0 / 94.0
                elif "min" in key:
                    record_step[key] = value * 28.0 / 94.0

        self.log_dict(
            record_step,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            add_dataloader_idx=False,
        )

        if batch_idx == 0 and self.hparams.test.num_id_to_save > 0:
            samples_original_scale_to_save = samples_original_scale[
                : self.hparams.test.num_id_to_save,
            ]
            samples_original_scale_to_save = samples_original_scale_to_save.cpu().numpy()
            save_dir = os.path.expanduser(self.hparams.test.save_dir)
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            np.save(
                f"{os.path.expanduser(self.hparams.test.save_dir)}/samples.npy",
                samples_original_scale_to_save,
            )


if __name__ == "__main__":
    from omegaconf import OmegaConf

    config = OmegaConf.load("configs/nba50/causal_scenario_prediction.yaml")
    interface = CausalScenarioPredictionInterface(config)
    x = torch.rand(10, 20, 11, 2)
    samples = interface.sample(x, max_length=5, num_paths=3)
    print(samples.shape)
