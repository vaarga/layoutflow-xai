import random
import os
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate, call
import rootutils
from collections import OrderedDict as OD
import torch
import numpy as np
from pytorch_fid.fid_score import calculate_frechet_distance
from tqdm import tqdm

from utils.fid_model import LayoutNet
from utils.metrics import compute_alignment, compute_overlap, compute_overlap_ignore_bg, compute_maximum_iou
from utils.utils import convert_bbox
from utils.visualization import draw_layout, visualize_trajectory

rootutils.setup_root(__file__, indicator=".git", pythonpath=True)

def set_seed(seed: int):
    """Sets the seed for reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Global seed set to: {seed}")

def get_data(dataloader, cfg):
    bbox = []
    label = []
    mask_bb = []

    bbox_for_miou = []
    for batch in dataloader:
        bbox.append(batch['bbox'])
        label.append(batch['type'])
        for i, L in enumerate(batch['length']):
            gt_bbox = convert_bbox(batch['bbox'][i, :L], f'{cfg.data.format}->xywh')
            bbox_for_miou.append([gt_bbox.cpu(), batch['type'][i][:L].cpu()])
        mask_bb.append(batch['mask'].squeeze())
    bbox, label, mask_bb = torch.cat(bbox), torch.cat(label), torch.cat(mask_bb)
    ltrb_bbox = convert_bbox(bbox, f'{cfg.data.format}->ltrb')
    return bbox, ltrb_bbox, label, mask_bb, bbox_for_miou

def load_generated_bbox_data(cfg, device):
    data = torch.load(cfg.load_bbox).to(cfg.device)
    print(f"Loaded {len(data)} samples from {cfg.load_bbox}")
    if 'uncond' in cfg.task:
        data = data[:2000]
    bbox, label, pad_mask = data[:, :, :4].to(torch.float32), data[:, :, 4].to(torch.int32), data[:, :, 5].to(
        torch.bool)
    ltrb_bbox = convert_bbox(bbox, f'{cfg.data.format}->ltrb')
    bbox = convert_bbox(bbox, f'{cfg.data.format}->xywh')
    ltrb_bbox, label, pad_mask = ltrb_bbox.to(device), label.to(device), pad_mask.to(device)
    bbox_for_miou = []
    for bb, cat, mask in zip(bbox, label, pad_mask):
        L = torch.sum(mask)
        bbox_for_miou.append([bb[:L].clone().cpu(), cat[:L].cpu()])
    return bbox, ltrb_bbox, label, pad_mask, bbox_for_miou


def should_pin_memory() -> bool:
    return torch.cuda.is_available()


@hydra.main(version_base=None, config_path="../conf", config_name="test.yaml")
def main(cfg: DictConfig):
    seed = getattr(cfg, "seed", 42)
    set_seed(seed)

    with torch.no_grad():
        device = torch.device(cfg.device)
        pin_memory = should_pin_memory()
        instance_idx = getattr(cfg, "instance_idx", -1)
        run_all = (instance_idx == -1)
        do_explain = getattr(cfg, "explain", False)

        val_loader = instantiate(cfg.dataset, dataset={'split': 'validation', 'lex_order': cfg.lex_order},
                                 shuffle=False, batch_size=1024, pin_memory=pin_memory)
        test_loader = instantiate(cfg.dataset, dataset={'split': 'test', 'lex_order': cfg.lex_order},
                                  shuffle=False, batch_size=1 if not run_all else 1024, pin_memory=pin_memory)

        if 'RICO' in cfg.experiment.expname:
            fid_model = LayoutNet(25, 20)
            state_dict = torch.load('./pretrained/fid_rico.pth.tar', map_location='cpu')
            length_dist = torch.tensor([0.0000, 0.0483, 0.0351, 0.0504, 0.0619, 0.0614, 0.0700, 0.0727, 0.0636, 0.0625,
                                        0.0636, 0.0552, 0.0523, 0.0400, 0.0461, 0.0534, 0.0373, 0.0333, 0.0322, 0.0276,
                                        0.0333])
        else:
            fid_model = LayoutNet(5, 20)
            state_dict = torch.load('./pretrained/fid_publaynet.pth.tar', map_location='cpu')
            length_dist = torch.tensor([0.0000, 0.0032, 0.0334, 0.0423, 0.0422, 0.0540, 0.0723, 0.0825, 0.0905, 0.0950,
                                        0.0959, 0.0895, 0.0781, 0.0620, 0.0478, 0.0359, 0.0262, 0.0188, 0.0140, 0.0097,
                                        0.0066])
        state = OD([(key.split("module.")[-1], state_dict[key]) for key in state_dict])
        fid_model.to(device)
        fid_model.load_state_dict(state)
        fid_model.requires_grad_(False)
        fid_model.eval()

        if not cfg.load_bbox:
            print("Loading Model...")
            model = hydra.utils.get_class(cfg.model._target_).load_from_checkpoint(
                cfg.checkpoint, map_location=device, weights_only=False
            )
            model.cond = cfg.cond_mask
            print(f'Conditioning mask: {model.cond}')
            print(f'Task: {cfg.task}')
            if "Flow" in cfg.model._target_:
                model.ode_solver = cfg.ode_solver
                print(f'ODE Solver: {cfg.ode_solver}')
            else:
                model.DM_model = instantiate(cfg.DM_model)
                print(f'DM Solver: {cfg.DM_model._target_}')
            model.inference_steps = cfg.model.inference_steps
            print(f'Inference steps: {model.inference_steps}')
            model = model.to(device)
            model.eval()

        if run_all:
            if 'RICO' in cfg.experiment.expname:
                data = torch.load('./pretrained/rico_test.pt')
                ltrb_bbox_test, label_test, mask_bb_test = data[..., :4], data[..., 4].long(), data[..., 5].bool()
                bbox_test = convert_bbox(ltrb_bbox_test, 'ltrb->xywh')
                gt = [[bb[:m.sum()], lab[:m.sum()]] for bb, lab, m in zip(bbox_test, label_test, mask_bb_test)]
            else:
                bbox_test, ltrb_bbox_test, label_test, mask_bb_test, gt = get_data(test_loader, cfg)
            _, ltrb_bbox_val, label_val, mask_bb_val, _ = get_data(val_loader, cfg)
            feats_real = fid_model.extract_features(ltrb_bbox_test.to(device), label_test.to(device),
                                                    (~mask_bb_test).to(device))
            mu1 = np.mean(feats_real.cpu().numpy(), axis=0)
            cov1 = np.cov(feats_real.cpu().numpy(), rowvar=False)
            feats_val = fid_model.extract_features(ltrb_bbox_val.to(device), label_val.to(device),
                                                   (~mask_bb_val).to(device))
            mu_val = np.mean(feats_val.cpu().numpy(), axis=0)
            cov_val = np.cov(feats_val.cpu().numpy(), rowvar=False)
            alignment_score = compute_alignment(bbox_test, mask_bb_test, format=cfg.data.format)
            if 'RICO' in cfg.experiment.expname:
                overlap_score = compute_overlap_ignore_bg(bbox_test, label_test, mask_bb_test, format=cfg.data.format)
            else:
                overlap_score = compute_overlap(bbox_test, mask_bb_test, format=cfg.data.format)
            fid_score = calculate_frechet_distance(mu1, cov1, mu_val, cov_val)
            print(
                f"[Sanity check] FID: {fid_score:.4f} | Align: {100 * alignment_score:.4f} | Overlap: {overlap_score:.4f}")

        metrics = {'fid': [], 'alignment': [], 'overlap': [], 'miou': []}
        run_num = 10 if cfg.multirun and not cfg.load_bbox else 1

        for n in range(run_num):
            if cfg.load_bbox:
                bbox, ltrb_bbox, label, pad_mask, bbox_for_miou = load_generated_bbox_data(cfg, device)
            else:
                bbox, label, pad_mask, bbox_for_miou = [], [], [], []

                # --- GENERATION LOOP ---
                for i, batch in enumerate(tqdm(test_loader, disable=(not run_all))):
                    if not run_all:
                        if i != instance_idx:
                            continue

                    batch['type'] = batch['type'].to(device)
                    batch['bbox'] = batch['bbox'].to(device)
                    batch['mask'] = batch['mask'].to(device)

                    if cfg.task == 'uncond':
                        batch['length'] = torch.multinomial(length_dist, num_samples=len(batch['length']),
                                                            replacement=True)
                    if cfg.task == 'refinement':
                        batch['bbox'] += torch.normal(0, std=0.01, size=batch['bbox'].size()).to(device)

                    if do_explain and not run_all:
                        try:
                            # Provide IG target settings to the model (from cfg)
                            target_idx = int(getattr(cfg, "target_idx", 0))
                            target_attr = str(getattr(cfg, "target_attr", "position"))

                            model.target_idx = target_idx
                            model.target_attr = target_attr

                            # Run inference with integrated gradients
                            geom_traj, cat_traj, cont_cat_traj, influence = model.inference(batch, task=cfg.task,
                                                                                            ig=True)

                            print("geom_traj:", geom_traj.shape)  # [T, B, N, 4]
                            print("cat_traj:", cat_traj.shape)  # [T, B, N]
                            print("influence:", influence.shape)  # [T, B, N, 3]

                            # Keep pipeline compatible: use final timestep for generated layout
                            geom_pred = geom_traj[-1]
                            cat_pred = cat_traj[-1]

                            if cfg.visualize:
                                visualize_trajectory(
                                    cfg=cfg,
                                    batch=batch,
                                    full_geom_pred=geom_traj,
                                    full_cat_pred=cat_traj,
                                    instance_idx=instance_idx,
                                    influence=influence,
                                    target_idx=target_idx,
                                    target_attr=target_attr,
                                )

                        except Exception as e:
                            print(f"[XAI] Error: {e}. Falling back to standard inference.")
                            geom_pred, cat_pred = model.inference(batch, task=cfg.task)

                    else:
                        # Standard Inference
                        geom_pred, cat_pred = model.inference(batch, task=cfg.task)

                    # --- Collect Results (Common Path) ---
                    bbox.append(geom_pred)
                    label.append(cat_pred)
                    pad_maski = torch.zeros(geom_pred.shape[:2], device=device, dtype=bool)
                    for k, L in enumerate(batch['length']):
                        pad_maski[k, :L] = True
                    pad_mask.append(pad_maski)

                    if not run_all:
                        break
                    if cfg.small:
                        break

                bbox = convert_bbox(torch.cat(bbox), f'{cfg.data.format}->xywh')
                ltrb_bbox, label, pad_mask = convert_bbox(bbox, 'xywh->ltrb'), torch.cat(label), torch.cat(pad_mask)
                for bb, cat, mask in zip(bbox, label, pad_mask):
                    L = torch.sum(mask)
                    bbox_for_miou.append([bb[:L].clone().cpu(), cat[:L].cpu()])

                results = torch.cat([bbox, label.unsqueeze(-1), pad_mask.unsqueeze(-1)], dim=-1)

                if run_all:
                    fname = f'{cfg.checkpoint.split("/")[-1][:-5]}_{cfg.task}_bbox.pt'
                    torch.save(results, f'./results/{fname}')
                    print(f'Results were saved at: ./results/{fname}')

            if cfg.task == 'uncond':
                bbox, ltrb_bbox, label, pad_mask = bbox[:2000], ltrb_bbox[:2000], label[:2000], pad_mask[:2000]
                bbox_for_miou = bbox_for_miou[:2000]

            if not run_all:
                print(f"Number of samples used: {len(label)} (Generated)")
                # Skip metrics for single instance
                break
            else:
                print(f"Number of samples: {len(label)} (Generated) / {len(label_test)} (Test)")

            feats_fake = fid_model.extract_features(ltrb_bbox, label, (~pad_mask))
            mu2 = np.mean(feats_fake.cpu().numpy(), axis=0)
            cov2 = np.cov(feats_fake.cpu().numpy(), rowvar=False)
            fid_score = calculate_frechet_distance(mu1, cov1, mu2, cov2)
            alignment_score = compute_alignment(bbox.cpu(), pad_mask.cpu())
            if 'RICO' in cfg.experiment.expname:
                overlap_score = compute_overlap_ignore_bg(bbox.cpu(), label.cpu(), pad_mask.cpu())
            else:
                overlap_score = compute_overlap(bbox.cpu(), pad_mask.cpu())
            max_iou = compute_maximum_iou(gt, bbox_for_miou) if cfg.calc_miou else -1
            print(
                f"FID: {fid_score:.4f} | Align: {100 * alignment_score:.4f} | Overlap: {overlap_score:.4f} | mIoU: {max_iou:.4f}")

            metrics['fid'].append(fid_score)
            metrics['alignment'].append(alignment_score)
            metrics['overlap'].append(overlap_score)
            metrics['miou'].append(max_iou)

    if cfg.visualize:
        os.makedirs('./vis', exist_ok=True)
        for i in range(min(20, len(bbox))):
            L = torch.sum(pad_mask[i]).long()
            fname = f'./vis/{instance_idx}.png' if not run_all else f'./vis/{i}.png'
            draw_layout(bbox[i, :L], label[i, :L], num_colors=26, square=False).save(fname)

    if cfg.multirun:
        fid_score = np.array(metrics['fid'])
        alignment_score = np.array(metrics['alignment'])
        overlap_score = np.array(metrics['overlap'])
        miou_score = np.array(metrics['miou'])
        print(f"Mean: FID: {fid_score.mean():.4f} | Align: {100 * alignment_score.mean():.4f} | " \
              f"Overlap: {overlap_score.mean():.4f} | mIoU: {miou_score.mean():.4f}")


if __name__ == "__main__":
    main()
