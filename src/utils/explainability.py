from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np
import torch
from captum.attr import IntegratedGradients


def load_ig_stats(stats_path: str) -> dict:
    with open(stats_path, "r") as f:
        return json.load(f)


@torch.no_grad()
def build_null_elem_data_from_stats(
    model,
    stats: dict,
    device,
    dtype,
    use_bbox: str = "median",  # "median" or "mean"
    use_type: str = "mode",    # "mode" or "mean"
) -> torch.Tensor:
    """
    Returns a single "null element" in DATA space, shape [1,1,D_data],
    where D_data matches the layout feature dimension (bbox + type encoding).
    DATA space means values are in [0,1], suitable for sampler.preprocess().
    """
    if use_bbox == "median":
        x0 = float(stats["x_median"])
        y0 = float(stats["y_median"])
        w0 = float(stats["w_median"])
        h0 = float(stats["h_median"])
    elif use_bbox == "mean":
        x0 = float(stats["x_mean"])
        y0 = float(stats["y_mean"])
        w0 = float(stats["w_mean"])
        h0 = float(stats["h_mean"])
    else:
        raise ValueError("use_bbox must be 'median' or 'mean'")

    bbox = torch.tensor([x0, y0, w0, h0], device=device, dtype=dtype).view(1, 1, 4)

    # --- Type part ---
    attr_encoding = getattr(model, "attr_encoding", None)
    num_cat = int(getattr(model, "num_cat", 1))

    if attr_encoding == "AnalogBit":
        analog_bit = getattr(model, "analog_bit", None)
        if analog_bit is None:
            raise RuntimeError("model.attr_encoding is AnalogBit but model.analog_bit is missing.")

        if use_type == "mode":
            t_idx = int(stats["type_mode"])
            t_idx_t = torch.tensor([[t_idx]], device=device, dtype=torch.long)  # [1,1]
            type_data = analog_bit.encode(t_idx_t).to(dtype=dtype)  # [1,1,attr_dim] in {0,1}
        elif use_type == "mean":
            hist = torch.tensor(stats["type_hist"], device=device, dtype=torch.float32)  # [K]
            denom = hist.sum().clamp_min(1.0)
            K = hist.numel()
            idxs = torch.arange(K, device=device, dtype=torch.long).view(K, 1)  # [K,1]
            bits = analog_bit.encode(idxs).to(dtype=torch.float32).squeeze(1)   # [K,attr_dim]
            bit_mean = (hist[:, None] * bits).sum(dim=0) / denom                # [attr_dim]
            type_data = bit_mean.to(dtype=dtype).view(1, 1, -1)
        else:
            raise ValueError("use_type must be 'mode' or 'mean'")
    else:
        # Scalar type encoding in [0,1]
        if use_type == "mode":
            t_idx = int(stats["type_mode"])
            t0 = t_idx / float(max(num_cat - 1, 1))
        elif use_type == "mean":
            # expected to already be normalized in [0,1]
            t0 = float(stats.get("conv_type_mean", 0.5))
        else:
            raise ValueError("use_type must be 'mode' or 'mean'")

        type_data = torch.tensor([t0], device=device, dtype=dtype).view(1, 1, 1)

    return torch.cat([bbox, type_data], dim=-1)  # [1,1,4+type_dim]


def resolve_ig_target(model, batch, dataset_name: str) -> tuple[
    int, int, tuple[int] | tuple[int, int] | tuple[int, int, int, int] | tuple[int, int, int, int, int]
]:
    instance_idx = getattr(model, "instance_idx", None)
    target_idx = getattr(model, "target_idx", None)
    target_attr = getattr(model, "target_attr", None)

    if instance_idx is None and isinstance(batch, dict) and ("instance_idx" in batch):
        instance_idx = str(batch["instance_idx"])
    if target_idx is None and isinstance(batch, dict) and ("target_idx" in batch):
        target_idx = int(batch["target_idx"])
    if target_attr is None and isinstance(batch, dict) and ("target_attr" in batch):
        target_attr = str(batch["target_attr"])

    if instance_idx is None:
        raise ValueError(
            "[IG] instance_idx not set. Set model.instance_idx or provide batch['instance_idx']."
        )
    if target_idx is None:
        raise ValueError(
            "[IG] target_idx not set. Set model.target_idx or provide batch['target_idx']."
        )
    if target_attr is None:
        raise ValueError(
            "[IG] target_attr not set. Set model.target_attr or provide batch['target_attr']."
        )

    ta = target_attr.lower().strip()
    if ta in ["pos", "position", "xy"]:
        out_dims = (0, 1)
    elif ta in ["size", "wh"]:
        out_dims = (2, 3)
    elif ta in ["x"]:
        out_dims = (0,)
    elif ta in ["y"]:
        out_dims = (1,)
    elif ta in ["w", "width"]:
        out_dims = (2,)
    elif ta in ["h", "height"]:
        out_dims = (3,)
    elif ta in ["geometry"]:
        out_dims = (0, 1, 2, 3)
    elif ta in ["category"]:
        if dataset_name == 'RICO':
            out_dims = (4, 5, 6, 7, 8)
        else:
            out_dims = (4, 5, 6)
    else:
        raise ValueError(
            f"[IG] Unsupported target_attr='{target_attr}'. "
            "Use: position|size|geometry|x|y|w|h|category"
        )

    return int(instance_idx), int(target_idx), out_dims


def compute_ig_influence_per_timestamp(
    model,
    batch,
    traj_pre: torch.Tensor,   # [T,B,N,D] preprocessed space states (the states you attribute at)
    t_span: torch.Tensor,     # [T] or [T,B] (both supported)
    cond_x: torch.Tensor,     # [B,N,D] preprocessed conditioning sample
    cond_mask: torch.Tensor,  # [B,N,D] 0/1 mask (1 = free, 0 = conditioned)
    dataset_name: str,
    influence_mode: str,
    out_dir: str,
) -> torch.Tensor:
    """
    Default:
      influence: [T,B,N,3] (pos,size,type) OR [T,B,N,1] for influence_mode="grouped_all"
    Special:
      if influence_mode=="per_xy" AND target_attr resolves to (0,1),
      influence: [T,B,N,2] where last dim is per-element contribution to target x and target y outputs.
    """
    instance_idx, target_idx, out_dims = resolve_ig_target(model, batch, dataset_name)
    ig_return_xy = (influence_mode == "per_xy") and (out_dims == (0, 1))

    cond_mask_f = cond_mask.to(dtype=cond_x.dtype)

    def forward_func(x_free: torch.Tensor, cond_vals: torch.Tensor, t_scalar: torch.Tensor) -> torch.Tensor:
        # Reconstruct the exact model input under the conditioning mask.
        x_model = (1.0 - cond_mask_f) * cond_vals + cond_mask_f * x_free

        # Normalize t to a [B] vector (works for scalar, [1], or [B])
        if not torch.is_tensor(t_scalar):
            t_scalar = torch.tensor(t_scalar, device=x_model.device, dtype=x_model.dtype)
        else:
            t_scalar = t_scalar.to(device=x_model.device, dtype=x_model.dtype)

        B = x_model.shape[0]
        if t_scalar.dim() == 0:
            t_vec = t_scalar.repeat(B)
        else:
            t_vec = t_scalar.view(-1)
            if t_vec.numel() == 1:
                t_vec = t_vec.repeat(B)
            elif t_vec.numel() != B:
                t_vec = t_vec[:1].repeat(B)

        # Model output is [B,N,D] (vector field for flow, noise residual for diffusion).
        v = model(x_model, cond_mask, t_vec)

        out = 0.0
        for d in out_dims:
            out = out + v[:, target_idx, d]
        return out  # [B]

    # valid element mask [B,N]
    valid_elem_mask: Optional[torch.Tensor] = None
    if isinstance(batch, dict) and ("mask" in batch):
        m = batch["mask"]
        if m.dim() == 3:
            m = m[..., 0]
        valid_elem_mask = m.to(dtype=cond_x.dtype)

    # load null element stats
    file_path = os.path.join("stats", f"{dataset_name}.json")
    stats = load_ig_stats(file_path)

    null_elem_data = build_null_elem_data_from_stats(
        model=model,
        stats=stats,
        device=cond_x.device,
        dtype=cond_x.dtype,
        use_bbox="median",
        use_type="mode",
    )  # [1,1,D_data]
    null_elem_pre = model.sampler.preprocess(null_elem_data)  # [1,1,D_pre]

    ig_steps = getattr(model, "ig_steps")
    T = traj_pre.shape[0]
    influences = []

    delta_arr = np.zeros(T, dtype=np.float32)
    diff_arr = np.zeros(T, dtype=np.float32)
    rel_delta_arr = np.zeros(T, dtype=np.float32)

    if ig_return_xy:
        delta_arr_x = np.zeros(T, dtype=np.float32)
        diff_arr_x = np.zeros(T, dtype=np.float32)
        rel_delta_arr_x = np.zeros(T, dtype=np.float32)
        delta_arr_y = np.zeros(T, dtype=np.float32)
        diff_arr_y = np.zeros(T, dtype=np.float32)
        rel_delta_arr_y = np.zeros(T, dtype=np.float32)

    with torch.enable_grad():
        for k in range(T):
            x_k = traj_pre[k]  # [B,N,D]
            null = null_elem_pre.expand_as(x_k)

            baseline_x_free = (1.0 - cond_mask_f) * x_k + cond_mask_f * null
            baseline_cond = cond_mask_f * cond_x + (1.0 - cond_mask_f) * null

            t_k = t_span[k]

            if ig_return_xy:
                orig_out_dims = out_dims

                # X component
                out_dims = (0,)
                ig_x = IntegratedGradients(forward_func)
                (attr_x_pair, delta_x) = ig_x.attribute(
                    inputs=(x_k, cond_x),
                    baselines=(baseline_x_free, baseline_cond),
                    additional_forward_args=(t_k,),
                    n_steps=ig_steps,
                    return_convergence_delta=True,
                )
                attr_free_x, attr_cond_x = attr_x_pair
                attr_x = cond_mask_f * attr_free_x + (1.0 - cond_mask_f) * attr_cond_x
                Fx = forward_func(x_k, cond_x, t_k).detach()
                Fb = forward_func(baseline_x_free, baseline_cond, t_k).detach()
                diff_x = (Fx - Fb).abs()
                delta_arr_x[k] = delta_x.detach().abs().mean().item()
                diff_arr_x[k] = diff_x.mean().item()
                rel_delta_arr_x[k] = (delta_x.detach().abs() / (diff_x + 1e-6)).mean().item()
                imp_x = attr_x.sum(dim=-1)  # [B,N]

                # Y component
                out_dims = (1,)
                ig_y = IntegratedGradients(forward_func)
                (attr_y_pair, delta_y) = ig_y.attribute(
                    inputs=(x_k, cond_x),
                    baselines=(baseline_x_free, baseline_cond),
                    additional_forward_args=(t_k,),
                    n_steps=ig_steps,
                    return_convergence_delta=True,
                )
                attr_free_y, attr_cond_y = attr_y_pair
                attr_y = cond_mask_f * attr_free_y + (1.0 - cond_mask_f) * attr_cond_y
                Fy = forward_func(x_k, cond_x, t_k).detach()
                Fby = forward_func(baseline_x_free, baseline_cond, t_k).detach()
                diff_y = (Fy - Fby).abs()
                delta_arr_y[k] = delta_y.detach().abs().mean().item()
                diff_arr_y[k] = diff_y.mean().item()
                rel_delta_arr_y[k] = (delta_y.detach().abs() / (diff_y + 1e-6)).mean().item()
                imp_y = attr_y.sum(dim=-1)  # [B,N]

                out_dims = orig_out_dims

                infl_k = torch.stack([imp_x, imp_y], dim=-1)  # [B,N,2]
                if valid_elem_mask is not None:
                    infl_k = infl_k * valid_elem_mask.unsqueeze(-1)
                influences.append(infl_k)

            else:
                ig = IntegratedGradients(forward_func)
                (attr_pair, delta) = ig.attribute(
                    inputs=(x_k, cond_x),
                    baselines=(baseline_x_free, baseline_cond),
                    additional_forward_args=(t_k,),
                    n_steps=ig_steps,
                    return_convergence_delta=True,
                )
                attr_free, attr_cond = attr_pair
                attr = cond_mask_f * attr_free + (1.0 - cond_mask_f) * attr_cond  # [B,N,D]

                Fx = forward_func(x_k, cond_x, t_k).detach()
                Fb = forward_func(baseline_x_free, baseline_cond, t_k).detach()
                diff = (Fx - Fb).abs()

                delta_arr[k] = delta.detach().abs().mean().item()
                diff_arr[k] = diff.mean().item()
                rel_delta_arr[k] = (delta.detach().abs() / (diff + 1e-6)).mean().item()

                if influence_mode == "grouped_all":
                    elem_inf = torch.abs(attr).sum(dim=-1)  # [B,N]
                    infl_k = elem_inf.unsqueeze(-1)          # [B,N,1]
                else:
                    geom_dim = int(getattr(model, "geom_dim", 4))
                    pos_inf = torch.abs(attr[..., 0]) + torch.abs(attr[..., 1])
                    size_inf = torch.abs(attr[..., 2]) + torch.abs(attr[..., 3])
                    type_inf = torch.abs(attr[..., geom_dim:]).sum(dim=-1)
                    infl_k = torch.stack([pos_inf, size_inf, type_inf], dim=-1)  # [B,N,3]

                if valid_elem_mask is not None:
                    infl_k = infl_k * valid_elem_mask.unsqueeze(-1)

                influences.append(infl_k)

    if ig_return_xy:
        stats = {
            "x": {
                "delta_mean": float(delta_arr_x.mean()),
                "diff_mean": float(diff_arr_x.mean()),
                "rel_delta_mean": float(rel_delta_arr_x.mean()),
            },
            "y": {
                "delta_mean": float(delta_arr_y.mean()),
                "diff_mean": float(diff_arr_y.mean()),
                "rel_delta_mean": float(rel_delta_arr_y.mean()),
            },
        }
    else:
        stats = {
                "delta_mean": float(delta_arr.mean()),
                "diff_mean": float(diff_arr.mean()),
                "rel_delta_mean": float(rel_delta_arr.mean()),
        }

    stats["influences"] = [t.detach().cpu().tolist()[0] for t in influences]
    out_path = os.path.join(out_dir, f"{instance_idx}_{target_idx}.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    return torch.stack(influences, dim=0)  # [T,B,N,?]
