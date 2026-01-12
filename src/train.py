from typing import Any, Dict, List, Optional, Tuple
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate
import torch
import rootutils
import json
import os
from pathlib import Path

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor

torch.cuda.manual_seed_all(42975)
torch.set_float32_matmul_precision('medium')

rootutils.setup_root(__file__, indicator=".git", pythonpath=True)


@torch.no_grad()
def compute_train_layout_stats(
        loader,
        num_cat: int,
        max_batches: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Computes bbox/type stats over ALL valid elements in the loader.

    Expected batch fields:
      - batch["bbox"]: [B, N, 4], normalized to [0,1]
      - batch["type"]: [B, N] (int categories 0..num_cat-1)
      - batch["mask"]: [B, N, 1] or [B, N] (bool, True = valid element)
        If mask missing: assumes all elements valid.

    Returns python-native floats/lists suitable for YAML/JSON serialization.
    """

    bbox_chunks = []
    type_chunks = []
    total_valid = 0

    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break

        bbox = batch["bbox"]  # [B, N, 4]
        typ = batch["type"]  # [B, N] (usually int)

        m = batch.get("mask", None)
        if m is None:
            valid = torch.ones_like(typ, dtype=torch.bool)
        else:
            # mask may be [B,N,1] or [B,N]
            if m.dim() == 3:
                m = m[..., 0]
            valid = m.to(dtype=torch.bool)

        # Flatten valid elements
        bbox_v = bbox[valid]  # [K,4]
        typ_v = typ[valid]  # [K]

        if bbox_v.numel() == 0:
            continue

        bbox_chunks.append(bbox_v.detach().cpu())
        type_chunks.append(typ_v.detach().cpu())

        total_valid += int(valid.sum().item())

    if total_valid == 0:
        raise RuntimeError("No valid elements found in training loader (mask may be wrong).")

    bbox_all = torch.cat(bbox_chunks, dim=0).float()  # [M,4]
    type_all = torch.cat(type_chunks, dim=0).long()  # [M]

    # bbox stats
    bbox_mean = bbox_all.mean(dim=0)
    bbox_median = bbox_all.median(dim=0).values

    # type stats
    # mean of category index
    type_mean_idx = type_all.float().mean()
    # mean of normalized conv_type in [0,1]
    conv_type_mean = type_mean_idx / float(max(num_cat - 1, 1))

    # histogram / mode can be useful too
    hist = torch.bincount(type_all, minlength=num_cat)
    type_mode = int(torch.argmax(hist).item())

    out = {
        "num_valid_elems": int(total_valid),

        "bbox_mean": bbox_mean.tolist(),  # [x,y,w,h]
        "bbox_median": bbox_median.tolist(),  # [x,y,w,h]

        "x_mean": float(bbox_mean[0].item()),
        "y_mean": float(bbox_mean[1].item()),
        "w_mean": float(bbox_mean[2].item()),
        "h_mean": float(bbox_mean[3].item()),

        "x_median": float(bbox_median[0].item()),
        "y_median": float(bbox_median[1].item()),
        "w_median": float(bbox_median[2].item()),
        "h_median": float(bbox_median[3].item()),

        "type_mean_idx": float(type_mean_idx.item()),
        "conv_type_mean": float(conv_type_mean.item()),
        "type_mode": type_mode,
        "type_hist": hist.tolist(),
    }
    return out


@hydra.main(version_base=None, config_path="../conf", config_name="train.yaml")
def main(cfg: DictConfig):
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    # Initiate Logger
    logger = instantiate(cfg.logger) if not cfg.test else None

    lr_monitor = LearningRateMonitor(logging_interval='step')
    exp_id = logger.experiment.id if 'Wandb' in cfg.logger._target_ else logger.log_dir
    ckpt_callback = instantiate(cfg.checkpoint, dirpath=cfg.checkpoint.dirpath.format(exp_id=exp_id))
    trainer = instantiate(cfg.trainer, callbacks=[ckpt_callback, lr_monitor], logger=logger)
    if trainer.global_rank==0 and 'Wandb' in cfg.logger._target_:
        logger.experiment.config.update(OmegaConf.to_container(cfg, resolve=True))

    train_loader = instantiate(cfg.dataset)
    val_loader = instantiate(cfg.dataset, dataset={'split': 'validation'}, shuffle=False)

    model = instantiate(cfg.model, expname=cfg.experiment.expname, format=cfg.data.format)

    if cfg.calculate_stats:
        # --- Compute & save training stats for IG baseline ---
        stats = compute_train_layout_stats(train_loader, num_cat=model.num_cat)

        dir_name = 'stats'
        file_path = os.path.join(dir_name, f'{cfg.dataset_name}.json')

        os.makedirs(dir_name, exist_ok=True)

        out_path = Path.cwd() / file_path
        out_path.write_text(json.dumps(stats, indent=2))

        print(f'[STATS] Saved training stats to: {out_path}')
    else:
        trainer.fit(
            model=model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=cfg.ckpt_path
        )

    return model


if __name__ == "__main__":
    main()