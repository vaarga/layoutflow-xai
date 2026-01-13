import os
import seaborn as sns
import torch
import math
from PIL import Image, ImageDraw, ImageOps

from utils.utils import convert_bbox


def draw_layout(layout, features, num_colors=6, format='xywh', background_img=None, square=True):
    '''
    layout (S, 4): layout bbox given as (x, y, w, h) and in range (0, 1)
    features (S, 1): one-dimensional features that determine the color 
    '''
    colors = gen_colors(num_colors)

    if background_img:
        img = background_img
    else:
        if square:
            img = Image.new('RGB', (256, 256), color=(255, 255, 255))
        else:
            img = Image.new('RGB', (256, int(4/3*256)), color=(255, 255, 255))
    draw = ImageDraw.Draw(img, 'RGBA') 
    layout = torch.clip(layout, 0, 1)
    if format == 'ltwh':
        box = torch.stack([layout[:,0], layout[:,1], layout[:,2]+layout[:,0], layout[:,3]+layout[:,1]], dim=1)
    elif format == 'xywh':
        box = torch.stack([layout[:,0]-layout[:,2]/2, layout[:,1]-layout[:,3]/2, layout[:,0]+layout[:,2]/2, layout[:,1]+layout[:,3]/2], dim=1)
    elif format == 'ltrb':
        box = torch.stack([layout[:,0], layout[:,1], torch.maximum(layout[:,0], layout[:,2]), torch.maximum(layout[:,1], layout[:,3])], dim=1)
    else:
        print(f"Error: {format} format not supported.")
    box = 255*torch.clamp(box, 0, 1)

    for i in range(len(layout)):
        x1, y1, x2, y2 = box[i]
        if not square:
            y1 = int(4/3*y1)
            y2 = int(4/3*y2)
        cat = features[i]-1
        col = colors[cat] if 0 <= cat < len(colors) else [0, 0, 0]
        if cat < 0:
            continue
        draw.rectangle([x1, y1, x2, y2],
                        outline=tuple(col) + (200,),
                        fill=tuple(col) + (64,),
                        width=2)

    # Add border around image
    img = ImageOps.expand(img, border=2)
    return img


def draw_xai_layout_xy_vectors(
    layout,
    features,
    num_colors=6,
    format="xywh",
    background_img=None,
    square=True,
    influence_xy=None,   # [L, 2] = [dx_contrib, dy_contrib] (SIGNED)
    target_idx=None,
    *,
    border_width=2,
    arrow_width=2,
    arrow_max_len_px=40,     # longest arrow length (pixels)
    arrow_head_len_px=14,
    arrow_head_angle_deg=25,
    skip_target_arrow=True,
    global_max_mag=None,     # float; if provided -> consistent scaling across frames
    aa_overlay=True,         # enable AA for arrows/circle
    aa_scale=4,              # supersampling factor (3–6 is typical)
):
    """
    Draws layout rectangles (same coloring convention), then draws one arrow per element i:
      vector_i = (influence_xy[i,0], influence_xy[i,1])
    Arrow direction: sign of (dx,dy)
    Arrow length: magnitude sqrt(dx^2 + dy^2) scaled to arrow_max_len_px.

    Rectangles remain non-antialiased.
    Arrows + target marker are antialiased via supersampled overlay when aa_overlay=True.
    """
    colors = gen_colors(num_colors)

    # Background (keep base as RGB; we'll composite AA overlay later)
    if background_img is not None:
        img = background_img.copy()
        if img.mode != "RGB":
            img = img.convert("RGB")
    else:
        img = Image.new("RGB", (256, 256), color=(255, 255, 255)) if square else Image.new(
            "RGB", (256, int(4/3*256)), color=(255, 255, 255)
        )

    W, H = img.size
    draw = ImageDraw.Draw(img, "RGBA")

    # Tensors -> CPU
    layout_t = layout.detach().cpu() if torch.is_tensor(layout) else torch.tensor(layout)
    feats_t  = features.detach().cpu() if torch.is_tensor(features) else torch.tensor(features)
    layout_t = torch.clip(layout_t, 0, 1)

    # Influence
    if influence_xy is not None:
        infl_t = influence_xy.detach().cpu() if torch.is_tensor(influence_xy) else torch.tensor(influence_xy)
        if infl_t.dim() != 2 or infl_t.size(-1) != 2:
            raise ValueError(f"influence_xy must have shape [S,2], got {tuple(infl_t.shape)}")
    else:
        infl_t = None

    # Boxes
    if format == "ltwh":
        box = torch.stack([layout_t[:,0], layout_t[:,1], layout_t[:,2]+layout_t[:,0], layout_t[:,3]+layout_t[:,1]], dim=1)
    elif format == "xywh":
        box = torch.stack([layout_t[:,0]-layout_t[:,2]/2, layout_t[:,1]-layout_t[:,3]/2, layout_t[:,0]+layout_t[:,2]/2, layout_t[:,1]+layout_t[:,3]/2], dim=1)
    elif format == "ltrb":
        box = torch.stack([layout_t[:,0], layout_t[:,1], torch.maximum(layout_t[:,0], layout_t[:,2]), torch.maximum(layout_t[:,1], layout_t[:,3])], dim=1)
    else:
        raise ValueError(f"Error: {format} format not supported.")

    box = 255 * torch.clamp(box, 0, 1)

    S = int(layout_t.shape[0])
    if target_idx is not None:
        try:
            target_idx = int(target_idx)
        except Exception:
            target_idx = None
        if target_idx is not None and not (0 <= target_idx < S):
            target_idx = None

    # --- 1) Draw rectangles on base (no AA, as requested) ---
    for i in range(S):
        cat_raw = feats_t[i].item()
        cat = int(cat_raw) - 1
        if cat < 0:
            continue
        col = colors[cat] if 0 <= cat < len(colors) else [0, 0, 0]

        x1, y1, x2, y2 = box[i].tolist()
        if not square:
            y1 = (4/3) * y1
            y2 = (4/3) * y2

        draw.rectangle([x1, y1, x2, y2], fill=tuple(col) + (64,))
        draw.rectangle([x1, y1, x2, y2], outline=tuple(col) + (200,), width=int(border_width))

    # Helper: draw arrows + circle onto a given ImageDraw with coordinate multiplier
    def _draw_arrows_and_target(draw_obj, coord_mul: float):
        nonlocal infl_t, target_idx

        # Draw arrows
        if infl_t is not None and S > 0:
            dx = infl_t[:, 0]
            dy = infl_t[:, 1]
            mag = torch.sqrt(dx*dx + dy*dy)

            # scaling (use global max if provided for consistency across frames)
            max_mag = float(global_max_mag) if (global_max_mag is not None) else float(mag.max().item() if mag.numel() else 0.0)
            if max_mag <= 1e-12:
                max_mag = 1.0
            scale = float(arrow_max_len_px) / max_mag

            head_angle = math.radians(float(arrow_head_angle_deg))
            shaft_w = max(1, int(round(float(arrow_width) * coord_mul)))
            hl = float(arrow_head_len_px) * coord_mul

            for i in range(S):
                if skip_target_arrow and (target_idx is not None) and (i == target_idx):
                    continue

                m = float(mag[i].item())
                if m <= 1e-12:
                    continue

                # alpha by relative magnitude
                a = int(round(255.0 * min(1.0, m / max_mag)))
                a = max(40, a)  # keep visible

                # Arrow from element center
                x1, y1, x2, y2 = box[i].tolist()
                if not square:
                    y1 = (4/3) * y1
                    y2 = (4/3) * y2

                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0

                vx = float(dx[i].item()) * scale
                vy = float(dy[i].item()) * scale

                ex = cx + vx
                ey = cy + vy

                # scale coords for supersampling layer
                cx_s, cy_s = cx * coord_mul, cy * coord_mul
                ex_s, ey_s = ex * coord_mul, ey * coord_mul

                # main shaft
                draw_obj.line([cx_s, cy_s, ex_s, ey_s], fill=(0, 0, 0, a), width=shaft_w)

                # arrow head (two lines)
                ang = math.atan2(vy, vx)

                xh1 = ex_s - hl * math.cos(ang - head_angle)
                yh1 = ey_s - hl * math.sin(ang - head_angle)
                xh2 = ex_s - hl * math.cos(ang + head_angle)
                yh2 = ey_s - hl * math.sin(ang + head_angle)

                draw_obj.line([ex_s, ey_s, xh1, yh1], fill=(0, 0, 0, a), width=shaft_w)
                draw_obj.line([ex_s, ey_s, xh2, yh2], fill=(0, 0, 0, a), width=shaft_w)

        # Draw target marker (circle)
        if target_idx is not None and 0 <= target_idx < S:
            x1, y1, x2, y2 = box[target_idx].tolist()
            if not square:
                y1 = (4/3) * y1
                y2 = (4/3) * y2

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            # scale for layer
            cx_s, cy_s = cx * coord_mul, cy * coord_mul
            r = 3.0 * coord_mul
            w = max(1, int(round(2.0 * coord_mul)))

            draw_obj.ellipse([cx_s - r, cy_s - r, cx_s + r, cy_s + r],
                             outline=(0, 0, 0, 255),
                             fill=(0, 0, 0, 255),
                             width=w)

    # --- 2) Draw arrows + circle (AA overlay if enabled) ---
    if aa_overlay and (infl_t is not None or target_idx is not None):
        aa = int(max(1, aa_scale))
        # supersampled transparent overlay
        overlay_hi = Image.new("RGBA", (W * aa, H * aa), (0, 0, 0, 0))
        draw_hi = ImageDraw.Draw(overlay_hi, "RGBA")

        _draw_arrows_and_target(draw_hi, coord_mul=float(aa))

        # downsample overlay to base size with high-quality filter
        overlay = overlay_hi.resize((W, H), resample=Image.LANCZOS)

        # composite onto base
        base_rgba = img.convert("RGBA")
        base_rgba = Image.alpha_composite(base_rgba, overlay)
        img = base_rgba.convert("RGB")
    else:
        # fallback: draw directly on base (non-AA)
        _draw_arrows_and_target(draw, coord_mul=1.0)

    img = ImageOps.expand(img, border=2, fill=(255, 255, 255))
    return img


def draw_xai_layout(
    layout,
    features,
    num_colors=6,
    format="xywh",
    background_img=None,
    square=True,
    influence=None,      # [L, 3] = [pos, size, type] raw IG (can be negative)
    target_idx=None,     # int
    *,
    border_width=2,
    point_radius=3,
    star_outer_radius=7,
):
    """
    layout (S, 4): bbox in (0,1)
    features (S,): category ids used only for base color (no text shown)

    If influence is provided:
      - influence is converted to per-category percentages over all elements (per timestamp)
      - marker opacity encodes position influence %
      - border opacity encodes size influence %
      - fill opacity encodes type influence %
      - target_idx gets a STAR marker; others get DOT marker
    """
    colors = gen_colors(num_colors)

    # Create background image
    if background_img:
        img = background_img
    else:
        if square:
            img = Image.new("RGB", (256, 256), color=(255, 255, 255))
        else:
            img = Image.new("RGB", (256, int(4 / 3 * 256)), color=(255, 255, 255))

    draw = ImageDraw.Draw(img, "RGBA")

    # Convert inputs to CPU tensors (safe for PIL drawing)
    if torch.is_tensor(layout):
        layout_t = layout.detach().cpu()
    else:
        layout_t = torch.tensor(layout)

    if torch.is_tensor(features):
        feats_t = features.detach().cpu()
    else:
        feats_t = torch.tensor(features)

    layout_t = torch.clip(layout_t, 0, 1)

    # Prepare influence percentages if provided
    influence_pct = None
    if influence is not None:
        if torch.is_tensor(influence):
            infl_t = influence.detach().cpu()
        else:
            infl_t = torch.tensor(influence)

        # Expect shape [S, 3]
        if infl_t.dim() != 2 or infl_t.size(-1) != 3:
            raise ValueError(f"influence must have shape [S,3], got {tuple(infl_t.shape)}")

        # Influence power = magnitude (so negatives do not create negative opacity)
        power = infl_t.abs()  # does NOT modify original tensor

        # Per-category normalization over elements (sum to 1 per column)
        denom = power.sum(dim=0, keepdim=True)  # [1,3]
        influence_pct = torch.where(
            denom > 0,
            power / denom,
            torch.zeros_like(power),
        )  # [S,3] in [0,1]

    # Convert layout to pixel boxes
    if format == "ltwh":
        box = torch.stack(
            [layout_t[:, 0], layout_t[:, 1], layout_t[:, 2] + layout_t[:, 0], layout_t[:, 3] + layout_t[:, 1]],
            dim=1,
        )
    elif format == "xywh":
        box = torch.stack(
            [
                layout_t[:, 0] - layout_t[:, 2] / 2,
                layout_t[:, 1] - layout_t[:, 3] / 2,
                layout_t[:, 0] + layout_t[:, 2] / 2,
                layout_t[:, 1] + layout_t[:, 3] / 2,
            ],
            dim=1,
        )
    elif format == "ltrb":
        box = torch.stack(
            [
                layout_t[:, 0],
                layout_t[:, 1],
                torch.maximum(layout_t[:, 0], layout_t[:, 2]),
                torch.maximum(layout_t[:, 1], layout_t[:, 3]),
            ],
            dim=1,
        )
    else:
        raise ValueError(f"Error: {format} format not supported.")

    box = 255 * torch.clamp(box, 0, 1)

    # Validate target_idx
    S = int(layout_t.shape[0])
    if target_idx is not None:
        try:
            target_idx = int(target_idx)
        except Exception:
            target_idx = None
        if target_idx is not None and not (0 <= target_idx < S):
            target_idx = None

    def star_polygon(cx, cy, r_outer, r_inner, num_points=5):
        pts = []
        angle = -math.pi / 2  # start upwards
        step = math.pi / num_points
        for k in range(num_points * 2):
            r = r_outer if (k % 2 == 0) else r_inner
            x = cx + r * math.cos(angle)
            y = cy + r * math.sin(angle)
            pts.append((x, y))
            angle += step
        return pts

    # Draw elements
    for i in range(S):
        # Category color (kept from your original function)
        cat_raw = feats_t[i].item()
        cat = int(cat_raw) - 1  # keep your original convention
        if cat < 0:
            continue

        col = colors[cat] if 0 <= cat < len(colors) else [0, 0, 0]

        x1, y1, x2, y2 = box[i].tolist()

        # Handle non-square aspect
        if not square:
            y1 = (4 / 3) * y1
            y2 = (4 / 3) * y2

        # Influence-driven alphas
        if influence_pct is None:
            # Original behavior (backward compatible)
            outline_rgba = tuple(col) + (200,)
            fill_rgba = tuple(col) + (64,)
            marker_alpha = None
        else:
            pos_p = float(influence_pct[i, 0].item())   # [0,1]
            size_p = float(influence_pct[i, 1].item())  # [0,1]
            type_p = float(influence_pct[i, 2].item())  # [0,1]

            marker_alpha = int(round(pos_p * 255))
            border_alpha = int(round(size_p * 255))
            fill_alpha = int(round(type_p * 255))

            # Background encodes TYPE influence (opacity)
            fill_rgba = tuple(col) + (fill_alpha,)

            # Border encodes SIZE influence (opacity)
            outline_rgba = (0, 0, 0, border_alpha)

        # Draw background (type influence)
        draw.rectangle([x1, y1, x2, y2], fill=fill_rgba)

        # Draw border (size influence)
        draw.rectangle([x1, y1, x2, y2], outline=outline_rgba, width=int(border_width))

        # Draw center marker (position influence)
        if marker_alpha is not None:
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            if target_idx is not None and i == target_idx:
                # STAR
                pts = star_polygon(
                    cx, cy,
                    r_outer=float(star_outer_radius),
                    r_inner=float(star_outer_radius) * 0.5,
                    num_points=5,
                )
                draw.polygon(pts, fill=(0, 0, 0, marker_alpha))
            else:
                # POINT
                r = float(point_radius)
                draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(0, 0, 0, marker_alpha))

    # Keep your border expansion, but make it white so it doesn’t add “extra visuals”
    img = ImageOps.expand(img, border=2, fill=(255, 255, 255))
    return img


def gen_colors(num_colors):
    """
    Generate uniformly distributed `num_colors` colors
    :param num_colors:
    :return:
    """
    palette = sns.color_palette("husl", num_colors)
    rgb_triples = [[int(x[0]*255), int(x[1]*255), int(x[2]*255)] for x in palette]
    return rgb_triples


def visualize_trajectory(
        cfg,
        batch,
        full_geom_pred,
        full_cat_pred,
        instance_idx,
        influence,
        target_idx,
        target_attr,
        *,
        out_root="./vis_traj",
        num_colors=26,
        square=False,
        ig_return_xy=False,
) -> None:
    """
    Saves one animated GIF per instance showing the trajectory over T steps.
    Intended to be called only in the do_explain + not run_all path.
    """
    # Determine valid element count L (mirrors later usage with pad_mask)
    if torch.is_tensor(batch["length"]):
        L = int(batch["length"][0].item())
    else:
        L = int(batch["length"][0])

    # Convert trajectory bboxes to xywh because draw_xai_layout uses xywh
    traj_bbox_xywh = convert_bbox(full_geom_pred, f'{cfg.data.format}->xywh')

    # Output directory per instance
    out_dir = os.path.join(out_root, cfg.dataset_name, cfg.task, str(instance_idx))
    os.makedirs(out_dir, exist_ok=True)

    fps = 5.0
    duration_ms = int(round(1000.0 / fps))

    # Render frames
    frames = []
    frames_xai = []
    T = int(traj_bbox_xywh.shape[0])

    max_mag_global = None
    if ig_return_xy:
        infl_xy = influence[:, 0, :L].detach().cpu()  # [T, L, 2]
        mags = torch.sqrt(infl_xy[..., 0] ** 2 + infl_xy[..., 1] ** 2).reshape(-1)
        if mags.numel():
            max_mag_global = float(torch.quantile(mags, 0.95).item())  # robust scaling
            max_mag_global = max(max_mag_global, 1e-6)
        else:
            max_mag_global = 1.0

    for t in range(T):
        img = draw_layout(
            traj_bbox_xywh[t, 0, :L].detach().cpu(),
            full_cat_pred[t, 0, :L].detach().to(torch.long).cpu(),
            num_colors=num_colors,
            square=square,
        )
        if ig_return_xy:
            img_xai = draw_xai_layout_xy_vectors(
                traj_bbox_xywh[t, 0, :L].detach().cpu(),
                full_cat_pred[t, 0, :L].detach().to(torch.long).cpu(),
                num_colors=num_colors,
                square=square,
                format="xywh",
                influence_xy=influence[t, 0, :L].detach().cpu(),  # [L,2]
                target_idx=target_idx,
                global_max_mag=max_mag_global,
                skip_target_arrow=False,  # <-- ADD THIS
            )
        else:
            img_xai = draw_xai_layout(
                traj_bbox_xywh[t, 0, :L].detach().cpu(),
                full_cat_pred[t, 0, :L].detach().to(torch.long).cpu(),
                num_colors=num_colors,
                square=square,
                format="xywh",
                influence=influence[t, 0, :L].detach().cpu(),  # [L,3]
                target_idx=target_idx,
            )

        # Ensure GIF-compatible mode
        # (RGBA is okay, but many viewers handle palette GIFs better)
        frames.append(img.convert("P", palette=Image.ADAPTIVE))
        frames_xai.append(img_xai.convert("P", palette=Image.ADAPTIVE))

    if len(frames) != 0:
        gif_path = os.path.join(out_dir, f"default.gif")

        # Save as looping GIF
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,  # ms per frame
            loop=0,  # loop forever
            disposal=2,  # restore to background between frames (reduces artifacts)
            optimize=False,
        )

        print(f"[NORMAL] Saved trajectory GIF with {T} frames to: {gif_path}")

    if len(frames_xai) != 0:
        gif_path = os.path.join(out_dir, f"{target_idx}_{target_attr}.gif")

        # Save as looping GIF
        frames_xai[0].save(
            gif_path,
            save_all=True,
            append_images=frames_xai[1:],
            duration=duration_ms,  # ms per frame
            loop=0,  # loop forever
            disposal=2,  # restore to background between frames (reduces artifacts)
            optimize=False,
        )

        print(f"[XAI] Saved trajectory GIF with {T} frames to: {gif_path}")
