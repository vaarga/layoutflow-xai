from typing import Text
import torch
import torch.nn as nn
import numpy as np
import json
import os
from torchcfm import ConditionalFlowMatcher
from torchdyn.core import NeuralODE

from src.models.BaseGenModel import BaseGenModel
from src.utils.fid_calculator import FID_score
from src.utils.analog_bit import AnalogBit


def load_ig_stats(stats_path: str):
    with open(stats_path, "r") as f:
        stats = json.load(f)

    return stats


class LayoutFlow(BaseGenModel):
    def __init__(
            self,
            backbone_model,
            sampler,
            optimizer,
            scheduler=None,
            loss_fcn = 'mse', 
            data_path: str = '',
            format = 'xywh',
            sigma = 0.0,
            fid_calc_every_n = 20,
            expname = 'LayoutFlow',
            num_cat = 6,
            time_sampling = 'uniform',
            sample_padding=False,
            inference_steps = 100,
            ode_solver = 'euler',
            cond = 'uncond',
            attr_encoding = 'continuous',
            train_traj = 'linear',
            add_loss = '',
            add_loss_weight=1,
            mask_padding = False,
            cf_guidance=0,
        ):
        self.format = format
        self.init_dist = sampler.distribution
        self.time_sampling = time_sampling
        self.time_weight = float(time_sampling.split('_')[-1]) if 'late_focus_' in time_sampling else 0.3
        self.sample_padding = sample_padding
        self.fid_calc_every_n = fid_calc_every_n
        self.cond = cond
        self.attr_encoding = attr_encoding
        self.target_idx = None  # int
        self.target_attr = None  # str: "position" or "size" (also supports "x","y","w","h")
        self.ig_steps = 300  # number of IG steps per timestamp (trade-off speed vs quality)
        if attr_encoding == 'AnalogBit':
            self.analog_bit = AnalogBit(num_cat)
        if fid_calc_every_n != 0: 
            fid_model = FID_score(dataset=expname.split('_')[0], data_path=data_path, calc_every_n=fid_calc_every_n)
        else:
            fid_model=None
        super().__init__(data_path=data_path, optimizer=optimizer, scheduler=scheduler, expname=expname, fid_model=fid_model)

        self.attr_dim = int(np.ceil(np.log2(num_cat))) if attr_encoding == 'AnalogBit' else 1
        self.num_cat = num_cat
        self.model = backbone_model
        self.inference_steps = inference_steps
        self.ode_solver = ode_solver
        self.sampler = sampler
        self.mask_padding = mask_padding
        self.cf_guidance = cf_guidance

        # Training Parameters
        self.train_traj = train_traj
        self.FM = ConditionalFlowMatcher(sigma=sigma)
        self.loss_fcn = nn.MSELoss() if loss_fcn!='l1' else nn.L1Loss()
        self.add_loss = add_loss
        self.add_loss_weight = add_loss_weight
        self.save_hyperparameters(ignore=["backbone_model", "sampler"])

        
    def training_step(self, batch, batch_idx):
        # Conditioning Mask
        cond_mask = self.get_cond_mask(batch)
        # Obtain initial sample x_0 and data sample x_1
        x0, x1 = self.get_start_end(batch)
        # Sample timestep
        t = self.sample_t(x0)
        # Calculate intermediate sample x_t based on time and trajectory 
        xt, ut = self.sample_xt(batch, x0, x1, cond_mask, t)

        # Prediction of vector field using backbone model
        vt = self(xt, cond_mask, t.squeeze(-1))

        # Loss calculation
        loss = self.loss_fcn(cond_mask*vt, cond_mask*ut)
        if self.add_loss:
            loss = self.additional_losses(cond_mask, ut, vt, loss) 
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)

        return loss

    def sample_xt(self, batch, x0, x1, cond_mask, t):
        eps = self.FM.sample_noise_like(x0) if self.FM.sigma != 0 else 0
        if self.train_traj == 'sin':
            tpad = t.reshape(-1, *([1] * (x0.dim() - 1)))
            xt = (1 - torch.sin(tpad * torch.pi/2)) * x0 + torch.sin(tpad * torch.pi/2) * x1 + self.FM.sigma * eps
            ut = torch.pi/2 * torch.cos(tpad * torch.pi / 2) * (x1 - x0)
        elif self.train_traj == 'sincos':
            tpad = t.reshape(-1, *([1] * (x0.dim() - 1)))
            xt = torch.cos(tpad * torch.pi/2) * x0 + torch.sin(tpad * torch.pi/2) * x1 + self.FM.sigma * eps
            ut = torch.pi/2 * (torch.cos(tpad * torch.pi / 2) * x1 - torch.sin(tpad * torch.pi / 2) * x0)
        else: #linear
            xt = self.FM.sample_xt(x0, x1, t, eps) 
            ut = self.FM.compute_conditional_flow(x0, x1, t, xt)
        xt = (1-cond_mask) * x1 + cond_mask * xt
        if self.mask_padding:
            xt = batch['mask'] * xt - (~batch['mask']) * torch.ones_like(xt)
            ut *= batch['mask']

        return xt, ut

    def inference(self, batch, full_traj=False, task=None, ig: bool = False, dataset_name='', ig_return_xy=False):
        """
        If ig=True:
          - returns exactly as full_traj=True plus influence tensor at the end:
            (traj_geom, cat, cont_cat, influence)
          - influence shape: [T, B, N, 3] with last dim = [pos, size, type]
        """
        # Sample initial layout x_0
        x0 = self.sampler.sample(batch)

        # Get conditioning mask
        cond_mask = self.get_cond_mask(batch)

        # Get conditional sample
        if self.attr_encoding == 'AnalogBit':
            conv_type = self.analog_bit.encode(batch['type'])
        else:
            conv_type = batch['type'].unsqueeze(-1) / (self.num_cat - 1)

        ref = torch.cat([batch['bbox'], conv_type], dim=-1)
        cond_x = self.sampler.preprocess(ref)
        cond_x = batch['mask'] * cond_x + (~batch['mask']) * ref

        # Create model wrapper for NeuralODE
        if task == 'condinf':
            vector_field = cond_wrapper(self, cond_x, cond_mask, batch=batch)
        else:
            vector_field = torch_wrapper(self, cond_x, cond_mask, cf_guidance=self.cf_guidance)

        # Solve NeuralODE
        if self.ode_solver == 'euler':
            node = NeuralODE(vector_field, solver=self.ode_solver, sensitivity="adjoint")
        else:
            node = NeuralODE(vector_field, solver=self.ode_solver, sensitivity="adjoint", atol=1e-4, rtol=1e-4)

        # IMPORTANT: we do not need gradients through the ODE solve for IG here.
        # IG is computed on the backbone vector field per timestamp using the solved trajectory states.
        with torch.no_grad():
            if task == 'refinement':
                t_span = torch.linspace(0.97, 1, self.inference_steps, device=x0.device)
                traj_pre = node.trajectory(cond_x, t_span=t_span)
            else:
                t_span = torch.linspace(0, 1, self.inference_steps, device=x0.device)
                traj_pre = node.trajectory(x0, t_span=t_span)

        # traj_pre is in "preprocessed" space; decode to data space for outputs as before
        traj = self.sampler.preprocess(traj_pre, reverse=True)

        # Post-processing and decoding of obtained trajectory
        if self.attr_encoding == 'AnalogBit':
            cont_cat = self.analog_bit.decode(traj[..., self.geom_dim:])
        else:
            cont_cat = traj[..., -1] * (self.num_cat - 1) + 0.5

        cont_cat = (1 - cond_mask[:, :, -1]) * batch['type'] + cond_mask[:, :, -1] * cont_cat
        cat = torch.clip(cont_cat.to(torch.int), 0, self.num_cat - 1)

        traj = (1 - cond_mask[:, :, :self.geom_dim]) * batch['bbox'][None] + cond_mask[:, :, :self.geom_dim] * traj[..., :self.geom_dim]
        self.input_cond = [
            (1 - cond_mask[:, :, :self.geom_dim]) * batch['bbox'],
            (1 - cond_mask[:, :, -1]) * batch['type']
        ]

        # --------------------------
        # Integrated Gradients block
        # --------------------------
        if ig:
            influence = self.compute_ig_influence_per_timestamp(
                batch=batch,
                traj_pre=traj_pre,
                t_span=t_span,
                cond_x=cond_x,
                cond_mask=cond_mask,
                dataset_name=dataset_name,
                ig_return_xy=ig_return_xy
            )
            # Requirement: if ig=True, return exactly like full_traj=True plus influence
            return traj, cat, cont_cat, influence

        # Default behavior
        return (traj, cat, cont_cat) if full_traj else (traj[-1], cat[-1])

    @torch.no_grad()
    def build_null_elem_data_from_stats(
            self,
            stats: dict,
            device,
            dtype,
            use_bbox: str = "median",  # "median" or "mean"
            use_type: str = "mode",  # "mode" or "mean"
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
        if getattr(self, "attr_encoding", None) == "AnalogBit":
            # Determine a representative category index
            if use_type == "mode":
                t_idx = int(stats["type_mode"])
                t_idx_t = torch.tensor([[t_idx]], device=device, dtype=torch.long)  # [1,1]
                type_data = self.analog_bit.encode(t_idx_t).to(dtype=dtype)  # [1,1,attr_dim] in {0,1}
            elif use_type == "mean":
                # Weighted mean of AnalogBit bits under the empirical type histogram
                hist = torch.tensor(stats["type_hist"], device=device, dtype=torch.float32)  # [num_cat]
                denom = hist.sum().clamp_min(1.0)

                # Build [num_cat, attr_dim] encodings and take weighted average
                idxs = torch.arange(len(stats["type_hist"]), device=device, dtype=torch.long).view(-1, 1)  # [K,1]
                bits = self.analog_bit.encode(idxs).to(dtype=torch.float32).squeeze(1)  # [K,attr_dim]
                bit_mean = (hist[:, None] * bits).sum(dim=0) / denom  # [attr_dim]
                type_data = bit_mean.to(dtype=dtype).view(1, 1, -1)  # [1,1,attr_dim]
            else:
                raise ValueError("use_type must be 'mode' or 'mean'")

        else:
            # Scalar type encoding in [0,1]
            if use_type == "mode":
                t_idx = int(stats["type_mode"])
                t0 = t_idx / float(max(self.num_cat - 1, 1))
            elif use_type == "mean":
                t0 = float(stats["conv_type_mean"])  # already normalized in [0,1]
            else:
                raise ValueError("use_type must be 'mode' or 'mean'")

            type_data = torch.tensor([t0], device=device, dtype=dtype).view(1, 1, 1)

        null_elem_data = torch.cat([bbox, type_data], dim=-1)  # [1,1,4+type_dim]

        return null_elem_data

    def resolve_ig_target(self, batch):
        """
        Target element index and attribute are expected to be provided via:
          - self.target_idx, self.target_attr (recommended), set externally from cfg
        Optionally supports batch['target_idx'], batch['target_attr'] if you prefer.
        """
        target_idx = getattr(self, "target_idx", None)
        target_attr = getattr(self, "target_attr", None)

        if target_idx is None and isinstance(batch, dict) and ("target_idx" in batch):
            target_idx = int(batch["target_idx"])
        if target_attr is None and isinstance(batch, dict) and ("target_attr" in batch):
            target_attr = str(batch["target_attr"])

        if target_idx is None:
            raise ValueError(
                "[LayoutFlow IG] target_idx not set. Set model.target_idx = cfg.target_idx "
                "or add batch['target_idx']."
            )
        if target_attr is None:
            raise ValueError(
                "[LayoutFlow IG] target_attr not set. Set model.target_attr = cfg.target_attr "
                "or add batch['target_attr']."
            )

        # Normalize naming
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
        else:
            raise ValueError(
                f"[LayoutFlow IG] Unsupported target_attr='{target_attr}'. "
                "Use: position|size|geometry|x|y|w|h."
            )

        return int(target_idx), out_dims

    def compute_ig_influence_per_timestamp(
            self,
            batch,
            traj_pre: torch.Tensor,  # [T, B, N, D] in preprocessed space
            t_span: torch.Tensor,  # [T]
            cond_x: torch.Tensor,  # [B, N, D]
            cond_mask: torch.Tensor,  # [B, N, D]
            dataset_name: Text,
            ig_return_xy: bool
    ) -> torch.Tensor:
        """
        Default (existing) behavior:
          Returns influence: [T, B, N, 3] where last dim is:
            0: position importance (|x|+|y|)
            1: size importance (|w|+|h|)
            2: type importance (sum(|type_dims|))

        Optional 2x20 behavior (only when batch["ig_return_xy"] == True and target_attr is position/xy):
          Returns influence: [T, B, N, 2] where last dim is:
            0: total per-element importance on target x-velocity
            1: total per-element importance on target y-velocity
        """
        try:
            from captum.attr import IntegratedGradients
        except Exception as e:
            raise ImportError(
                "[LayoutFlow IG] captum is required for ig=True. "
                "Install with: pip install captum"
            ) from e

        target_idx, out_dims = self.resolve_ig_target(batch)

        # only meaningful for position targets!
        ig_return_xy = ig_return_xy and (out_dims == (0, 1))

        cond_mask_f = cond_mask.to(dtype=cond_x.dtype)

        def forward_func(x_free: torch.Tensor, cond_vals: torch.Tensor, t_scalar: torch.Tensor) -> torch.Tensor:
            x_model = (1.0 - cond_mask_f) * cond_vals + cond_mask_f * x_free

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

            v = self(x_model, cond_mask, t_vec)  # [B,N,D]

            out = 0.0
            for d in out_dims:
                out = out + v[:, target_idx, d]
            return out

        ig = IntegratedGradients(forward_func)

        valid_elem_mask = None
        if isinstance(batch, dict) and ("mask" in batch):
            m = batch["mask"]
            if m.dim() == 3:
                m = m[..., 0]
            valid_elem_mask = m.to(dtype=cond_x.dtype)  # [B, N]

        influences = []
        T = traj_pre.shape[0]

        # --- Existing delta arrays (3x20 path) ---
        delta_arr = np.zeros(T, dtype=np.float32)
        diff_arr = np.zeros(T, dtype=np.float32)
        rel_delta_arr = np.zeros(T, dtype=np.float32)

        # --- NEW: separate delta arrays for 2x20 path ---
        if ig_return_xy:
            delta_arr_x = np.zeros(T, dtype=np.float32)
            diff_arr_x = np.zeros(T, dtype=np.float32)
            rel_delta_arr_x = np.zeros(T, dtype=np.float32)

            delta_arr_y = np.zeros(T, dtype=np.float32)
            diff_arr_y = np.zeros(T, dtype=np.float32)
            rel_delta_arr_y = np.zeros(T, dtype=np.float32)

        file_path = os.path.join("stats", f'{dataset_name}.json')
        stats = load_ig_stats(file_path)

        null_elem_data = self.build_null_elem_data_from_stats(
            stats=stats,
            device=cond_x.device,
            dtype=cond_x.dtype,
            use_bbox="median",
            use_type="mode",
        )  # [1,1,D]

        null_elem_pre = self.sampler.preprocess(null_elem_data)  # [1,1,D]

        with torch.enable_grad():
            for k in range(traj_pre.shape[0]):
                x_k = traj_pre[k]  # [B,N,D]
                null = null_elem_pre.expand_as(x_k)  # [B,N,D]

                baseline_x_free = (1.0 - cond_mask_f) * x_k + cond_mask_f * null
                baseline_cond = cond_mask_f * cond_x + (1.0 - cond_mask_f) * null

                # --------------------------
                # NEW: 2x20 branch (x and y separately)
                # --------------------------
                if ig_return_xy:
                    orig_out_dims = out_dims

                    # ----- X component -----
                    out_dims = (0,)
                    (attributions_x, delta_x) = ig.attribute(
                        inputs=(x_k, cond_x),
                        baselines=(baseline_x_free, baseline_cond),
                        additional_forward_args=(t_span[k],),
                        n_steps=int(getattr(self, "ig_steps", 32)),
                        return_convergence_delta=True,
                    )
                    attr_free_x, attr_cond_x = attributions_x
                    attr_x = cond_mask_f * attr_free_x + (1.0 - cond_mask_f) * attr_cond_x  # [B,N,D]

                    Fx = forward_func(x_k, cond_x, t_span[k]).detach()
                    Fb = forward_func(baseline_x_free, baseline_cond, t_span[k]).detach()
                    diff_x = (Fx - Fb).abs()
                    delta_arr_x[k] = delta_x.detach().abs().mean().item()
                    diff_arr_x[k] = diff_x.detach().mean().item()
                    rel_delta_arr_x[k] = (delta_x.detach().abs() / (diff_x.detach() + 1e-6)).mean().item()

                    # total per-element signed contribution on target-x output
                    imp_x = attr_x.sum(dim=-1)  # [B,N]

                    # ----- Y component -----
                    out_dims = (1,)
                    (attributions_y, delta_y) = ig.attribute(
                        inputs=(x_k, cond_x),
                        baselines=(baseline_x_free, baseline_cond),
                        additional_forward_args=(t_span[k],),
                        n_steps=int(getattr(self, "ig_steps", 32)),
                        return_convergence_delta=True,
                    )
                    attr_free_y, attr_cond_y = attributions_y
                    attr_y = cond_mask_f * attr_free_y + (1.0 - cond_mask_f) * attr_cond_y  # [B,N,D]

                    Fy = forward_func(x_k, cond_x, t_span[k]).detach()
                    Fby = forward_func(baseline_x_free, baseline_cond, t_span[k]).detach()
                    diff_y = (Fy - Fby).abs()
                    delta_arr_y[k] = delta_y.detach().abs().mean().item()
                    diff_arr_y[k] = diff_y.detach().mean().item()
                    rel_delta_arr_y[k] = (delta_y.detach().abs() / (diff_y.detach() + 1e-6)).mean().item()

                    # total per-element signed contribution on target-y output
                    imp_y = attr_y.sum(dim=-1)  # [B,N]

                    # restore
                    out_dims = orig_out_dims

                    infl_k = torch.stack([imp_x, imp_y], dim=-1)  # [B,N,2]

                    if valid_elem_mask is not None:
                        infl_k = infl_k * valid_elem_mask.unsqueeze(-1)

                    influences.append(infl_k)
                else:
                    # --------------------------
                    # Existing 3x20 branch (UNCHANGED)
                    # --------------------------
                    (attributions, delta) = ig.attribute(
                        inputs=(x_k, cond_x),
                        baselines=(baseline_x_free, baseline_cond),
                        additional_forward_args=(t_span[k],),
                        n_steps=int(getattr(self, "ig_steps", 32)),
                        return_convergence_delta=True,
                    )

                    attr_free, attr_cond = attributions
                    attr = cond_mask_f * attr_free + (1.0 - cond_mask_f) * attr_cond  # [B,N,D]

                    Fx = forward_func(x_k, cond_x, t_span[k]).detach()
                    Fb = forward_func(baseline_x_free, baseline_cond, t_span[k]).detach()
                    diff = (Fx - Fb).abs()
                    delta_s = delta.detach().abs().mean().item()
                    diff_s = diff.detach().mean().item()
                    rel_s = (delta.detach().abs() / (diff.detach() + 1e-6)).mean().item()

                    delta_arr[k] = delta_s
                    diff_arr[k] = diff_s
                    rel_delta_arr[k] = rel_s

                    pos_inf = torch.abs(attr[..., 0]) + torch.abs(attr[..., 1])  # [B,N]
                    size_inf = torch.abs(attr[..., 2]) + torch.abs(attr[..., 3])  # [B,N]
                    type_inf = torch.abs(attr[..., self.geom_dim:]).sum(dim=-1)  # [B,N]

                    infl_k = torch.stack([pos_inf, size_inf, type_inf], dim=-1)  # [B,N,3]

                    if valid_elem_mask is not None:
                        infl_k = infl_k * valid_elem_mask.unsqueeze(-1)

                    influences.append(infl_k)

        # Prints
        if ig_return_xy:
            print("X: delta mean:", float(delta_arr_x.mean()))
            print("X: diff mean:", float(diff_arr_x.mean()))
            print("X: rel_delta mean:", float(rel_delta_arr_x.mean()))
            print("Y: delta mean:", float(delta_arr_y.mean()))
            print("Y: diff mean:", float(diff_arr_y.mean()))
            print("Y: rel_delta mean:", float(rel_delta_arr_y.mean()))
        else:
            print("delta mean:", float(delta_arr.mean()))
            print("diff mean:", float(diff_arr.mean()))
            print("rel_delta mean:", float(rel_delta_arr.mean()))

        influence = torch.stack(influences, dim=0)  # [T, B, N, 3] or [T, B, N, 2]

        return influence


class torch_wrapper(torch.nn.Module):
    '''
    Wraps model to torchdyn compatible format.
    forward method defines a single step of the ODE solver.
    '''

    def __init__(self, model, cond_x, cond_mask=None, inverse=False, cf_guidance=0):
        super().__init__()
        self.model = model
        self.cond_x = cond_x
        self.cond_mask = cond_mask
        self.sign = -1 if inverse else 1
        self.cf_guidance = cf_guidance
        if cf_guidance:
            self.uncond_mask = torch.ones_like(self.cond_mask)

    def forward(self, t, x, *args, **kwargs):
        x = (1-self.cond_mask) * self.cond_x + self.cond_mask * x
        if self.sign == -1:
            t = 1 - t
        v = self.model(x, self.cond_mask, t.repeat(x.shape[0]))
        if self.cf_guidance:
            v = (1+self.cf_guidance) * v - self.cf_guidance * self.model(x, self.uncond_mask, t.repeat(x.shape[0]))
        return self.sign * v


class cond_wrapper(torch.nn.Module):
    '''
    Wraps model to torchdyn compatible format.
    This is an alternative conditioning method, that only uses the unconditional masking as described in the Appendix 
    of our paper (Section: Conditioning Analysis).
    '''

    def __init__(self, model, cond_x, cond_mask=None, batch=None):
        super().__init__()
        self.model = model
        self.cond_x = cond_x
        self.cond_mask = cond_mask
        self.batch = batch

    def forward(self, t, x, *args, **kwargs):
        v = self.batch['mask'] * self.model(x, torch.ones_like(self.cond_mask), t.repeat(x.shape[0]))
        cond_dir = (self.cond_x - x)
        new_v = self.cond_mask * v + (1-self.cond_mask) * cond_dir
        return new_v