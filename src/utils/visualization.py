import os
import seaborn as sns
import torch
import math
from PIL import Image, ImageDraw, ImageOps

from utils.utils import convert_bbox

DEFAULT_WIDTH = 2
MARKER_WIDTH_ADJ = 2


def gen_colors(num_colors):
    """
    Generate uniformly distributed `num_colors` colors
    :param num_colors:
    :return:
    """
    palette = sns.color_palette("husl", num_colors)
    rgb_triples = [[int(x[0]*255), int(x[1]*255), int(x[2]*255)] for x in palette]
    return rgb_triples


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
    arrow_max_len_px=40,     # longest arrow length (pixels)
    arrow_head_len_px=14,
    arrow_head_angle_deg=25,
    skip_target_arrow=True,
    global_max_mag=None,     # float; if provided -> consistent scaling across frames
    aa_overlay=True,         # enable AA for arrows/circle
    aa_scale=4,              # supersampling factor (3–6 is typical)
):
    """
    Rectangles are drawn on the base image (non-AA, as requested).
    Arrows + target circle are drawn on a supersampled transparent overlay (AA),
    then downsampled and composited onto the base.

    - No clamping of box corners to [0,1] (prevents "growing" illusion).
    - No outer padding/border is added.
    - Normalized coords are mapped to actual canvas size (W,H); overflow is clipped by PIL.
    """
    colors = gen_colors(num_colors)

    # --- Background ---
    if background_img is not None:
        img = background_img.copy()
        if img.mode != "RGB":
            img = img.convert("RGB")
    else:
        img = (
            Image.new("RGB", (256, 256), (255, 255, 255))
            if square
            else Image.new("RGB", (256, int(4 / 3 * 256)), (255, 255, 255))
        )

    W, H = img.size
    draw = ImageDraw.Draw(img, "RGBA")

    # --- Inputs -> tensors on CPU ---
    layout_t = layout.detach().cpu() if torch.is_tensor(layout) else torch.tensor(layout)
    feats_t = features.detach().cpu() if torch.is_tensor(features) else torch.tensor(features)

    # Do NOT clamp positions/corners; only guard against negative sizes
    if layout_t.numel() > 0 and layout_t.shape[-1] >= 4:
        layout_t = layout_t.clone()
        layout_t[:, 2:4] = torch.clamp(layout_t[:, 2:4], min=0.0)

    # Influence
    if influence_xy is not None:
        infl_t = influence_xy.detach().cpu() if torch.is_tensor(influence_xy) else torch.tensor(influence_xy)
        if infl_t.dim() != 2 or infl_t.size(-1) != 2:
            raise ValueError(f"influence_xy must have shape [S,2], got {tuple(infl_t.shape)}")
    else:
        infl_t = None

    S = int(layout_t.shape[0])

    # Target index sanitize
    if target_idx is not None:
        try:
            target_idx = int(target_idx)
        except Exception:
            target_idx = None
        if target_idx is not None and not (0 <= target_idx < S):
            target_idx = None

    # --- Boxes in normalized coords (NO CLAMP) ---
    if S == 0:
        box_n = torch.zeros((0, 4), dtype=torch.float32)
    else:
        if format == "ltwh":
            box_n = torch.stack(
                [
                    layout_t[:, 0],
                    layout_t[:, 1],
                    layout_t[:, 0] + layout_t[:, 2],
                    layout_t[:, 1] + layout_t[:, 3],
                ],
                dim=1,
            )
        elif format == "xywh":
            box_n = torch.stack(
                [
                    layout_t[:, 0] - layout_t[:, 2] / 2,
                    layout_t[:, 1] - layout_t[:, 3] / 2,
                    layout_t[:, 0] + layout_t[:, 2] / 2,
                    layout_t[:, 1] + layout_t[:, 3] / 2,
                ],
                dim=1,
            )
        elif format == "ltrb":
            box_n = torch.stack([layout_t[:, 0], layout_t[:, 1], layout_t[:, 2], layout_t[:, 3]], dim=1)
        else:
            raise ValueError(f"Error: {format} format not supported.")

    # Ensure correct ordering (robust against swapped corners)
    if S > 0:
        x1n = torch.minimum(box_n[:, 0], box_n[:, 2])
        x2n = torch.maximum(box_n[:, 0], box_n[:, 2])
        y1n = torch.minimum(box_n[:, 1], box_n[:, 3])
        y2n = torch.maximum(box_n[:, 1], box_n[:, 3])
        box_n = torch.stack([x1n, y1n, x2n, y2n], dim=1)

    # --- Convert to pixel coords based on actual canvas size ---
    sx = float(W - 1)
    sy = float(H - 1)
    box = torch.stack(
        [
            box_n[:, 0] * sx,
            box_n[:, 1] * sy,
            box_n[:, 2] * sx,
            box_n[:, 3] * sy,
        ],
        dim=1,
    ) if S > 0 else box_n

    # --- 1) Draw rectangles on base (non-AA) ---
    for i in range(S):
        cat_raw = feats_t[i].item()
        cat = int(cat_raw) - 1
        if cat < 0:
            continue
        col = colors[cat] if 0 <= cat < len(colors) else [0, 0, 0]

        x1, y1, x2, y2 = box[i].tolist()
        draw.rectangle([x1, y1, x2, y2], fill=tuple(col) + (64,))
        draw.rectangle([x1, y1, x2, y2], outline=tuple(col) + (200,), width=int(DEFAULT_WIDTH))

    # Helper: draw arrows + target circle onto a given ImageDraw, scaling coords by coord_mul
    def _draw_arrows_and_target(draw_obj, coord_mul: float):
        nonlocal infl_t, target_idx

        # Arrows
        if infl_t is not None and S > 0:
            dx = infl_t[:, 0]
            dy = infl_t[:, 1]
            mag = torch.sqrt(dx * dx + dy * dy)

            max_mag = float(global_max_mag) if (global_max_mag is not None) else float(mag.max().item() if mag.numel() else 0.0)
            if max_mag <= 1e-12:
                max_mag = 1.0

            scale = float(arrow_max_len_px) / max_mag

            head_angle = math.radians(float(arrow_head_angle_deg))
            shaft_w = max(1, int(round(float(DEFAULT_WIDTH) * coord_mul)))
            hl = float(arrow_head_len_px) * coord_mul

            for i in range(S):
                if skip_target_arrow and (target_idx is not None) and (i == target_idx):
                    continue

                m = float(mag[i].item())
                if m <= 1e-12:
                    continue

                a = int(round(255.0 * min(1.0, m / max_mag)))
                a = max(40, a)

                x1, y1, x2, y2 = box[i].tolist()
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0

                vx = float(dx[i].item()) * scale
                vy = float(dy[i].item()) * scale
                ex = cx + vx
                ey = cy + vy

                cx_s, cy_s = cx * coord_mul, cy * coord_mul
                ex_s, ey_s = ex * coord_mul, ey * coord_mul

                # shaft
                draw_obj.line([cx_s, cy_s, ex_s, ey_s], fill=(0, 0, 0, a), width=shaft_w)

                # head
                ang = math.atan2(vy, vx)
                xh1 = ex_s - hl * math.cos(ang - head_angle)
                yh1 = ey_s - hl * math.sin(ang - head_angle)
                xh2 = ex_s - hl * math.cos(ang + head_angle)
                yh2 = ey_s - hl * math.sin(ang + head_angle)

                draw_obj.line([ex_s, ey_s, xh1, yh1], fill=(0, 0, 0, a), width=shaft_w)
                draw_obj.line([ex_s, ey_s, xh2, yh2], fill=(0, 0, 0, a), width=shaft_w)

        # Target marker (filled circle)
        if target_idx is not None and 0 <= target_idx < S:
            x1, y1, x2, y2 = box[target_idx].tolist()
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            cx_s, cy_s = cx * coord_mul, cy * coord_mul
            marker_width = DEFAULT_WIDTH + MARKER_WIDTH_ADJ
            r = (marker_width / 2) * coord_mul

            draw_obj.ellipse(
                [cx_s - r, cy_s - r, cx_s + r, cy_s + r],
                fill=(0, 0, 0, 255),
            )

    # --- 2) AA overlay for arrows + circle ---
    if aa_overlay and (infl_t is not None or target_idx is not None):
        aa = int(max(1, aa_scale))
        overlay_hi = Image.new("RGBA", (W * aa, H * aa), (0, 0, 0, 0))
        draw_hi = ImageDraw.Draw(overlay_hi, "RGBA")

        _draw_arrows_and_target(draw_hi, coord_mul=float(aa))

        # Downsample and composite
        overlay = overlay_hi.resize((W, H), resample=Image.LANCZOS)
        base_rgba = img.convert("RGBA")
        img = Image.alpha_composite(base_rgba, overlay).convert("RGB")
    else:
        # Non-AA fallback
        _draw_arrows_and_target(draw, coord_mul=1.0)

    return img


def draw_xai_layout(
    layout,
    features,
    num_colors=6,
    format="xywh",
    background_img=None,
    square=True,
    influence=None,      # [S, 3] = [pos, size, type] raw IG (can be negative)
    target_idx=None,     # int
    *,
    influence_mode=None,
    t=None,
):
    """
    layout (S, 4): bbox in normalized coordinates (not forcibly clamped to [0,1])
    features (S,) or (S,1): category ids used only for base color

    If influence is provided:
      - influence is converted to per-element percentages over all elements (per timestamp)
      - marker opacity encodes position influence %
      - border opacity encodes size influence %
      - fill opacity encodes type influence %
      - target_idx gets a STAR marker; others get DOT marker

    IMPORTANT BEHAVIOR (per your request):
      - No clamping of box corners to [0,1] (prevents “growing/shrinking” illusion).
      - No outer padding/border is added.
      - Normalized coords are mapped to actual canvas size (W,H); overflow is naturally clipped by PIL.
    """
    colors = gen_colors(num_colors)

    # --- Background ---
    if background_img is not None:
        img = background_img.copy()
        if img.mode != "RGB":
            img = img.convert("RGB")
    else:
        if square:
            img = Image.new("RGB", (256, 256), color=(255, 255, 255))
        else:
            img = Image.new("RGB", (256, int(4 / 3 * 256)), color=(255, 255, 255))

    W, H = img.size
    draw = ImageDraw.Draw(img, "RGBA")

    # --- Inputs -> CPU tensors ---
    layout_t = layout.detach().cpu() if torch.is_tensor(layout) else torch.tensor(layout)
    feats_t = features.detach().cpu() if torch.is_tensor(features) else torch.tensor(features)

    # Do NOT clamp positions/corners; only guard against negative sizes (w,h)
    if layout_t.numel() > 0 and layout_t.shape[-1] >= 4:
        layout_t = layout_t.clone()
        layout_t[:, 2:4] = torch.clamp(layout_t[:, 2:4], min=0.0)

    # --- Prepare influence percentages if provided ---
    influence_pct = None
    if influence is not None:
        infl_t = influence.detach().cpu() if torch.is_tensor(influence) else torch.tensor(influence)

        if influence_mode == "grouped_all":
            if infl_t.dim() == 1:
                infl_t = infl_t.unsqueeze(-1)  # [S,1]
            if infl_t.dim() != 2 or infl_t.size(-1) != 1:
                raise ValueError(f"influence must have shape [S,1] for grouped_all, got {tuple(infl_t.shape)}")

            power = infl_t.abs()  # [S,1]
            denom = power.sum(dim=0, keepdim=True)  # [1,1]
            influence_pct = torch.where(denom > 0, power / denom, torch.zeros_like(power))  # [S,1] in [0,1]
        else:
            if infl_t.dim() != 2 or infl_t.size(-1) != 3:
                raise ValueError(f"influence must have shape [S,3], got {tuple(infl_t.shape)}")

            power = infl_t.abs()  # [S,3]
            denom = power.sum(dim=0, keepdim=True)  # [1,3]
            influence_pct = torch.where(denom > 0, power / denom, torch.zeros_like(power))  # [S,3] in [0,1]

    # --- Boxes in normalized coords (NO CLAMP) ---
    S = int(layout_t.shape[0])
    if S == 0:
        return img

    if format == "ltwh":
        box_n = torch.stack(
            [
                layout_t[:, 0],
                layout_t[:, 1],
                layout_t[:, 0] + layout_t[:, 2],
                layout_t[:, 1] + layout_t[:, 3],
            ],
            dim=1,
        )
    elif format == "xywh":
        box_n = torch.stack(
            [
                layout_t[:, 0] - layout_t[:, 2] / 2,
                layout_t[:, 1] - layout_t[:, 3] / 2,
                layout_t[:, 0] + layout_t[:, 2] / 2,
                layout_t[:, 1] + layout_t[:, 3] / 2,
            ],
            dim=1,
        )
    elif format == "ltrb":
        box_n = torch.stack(
            [
                layout_t[:, 0],
                layout_t[:, 1],
                layout_t[:, 2],
                layout_t[:, 3],
            ],
            dim=1,
        )
    else:
        raise ValueError(f"Error: {format} format not supported.")

    # Ensure correct ordering (robust if corners are swapped)
    x1n = torch.minimum(box_n[:, 0], box_n[:, 2])
    x2n = torch.maximum(box_n[:, 0], box_n[:, 2])
    y1n = torch.minimum(box_n[:, 1], box_n[:, 3])
    y2n = torch.maximum(box_n[:, 1], box_n[:, 3])
    box_n = torch.stack([x1n, y1n, x2n, y2n], dim=1)

    # --- Convert normalized coords to pixels using actual canvas size (W,H), no clamping ---
    sx = float(W - 1)
    sy = float(H - 1)
    box = torch.stack(
        [
            box_n[:, 0] * sx,
            box_n[:, 1] * sy,
            box_n[:, 2] * sx,
            box_n[:, 3] * sy,
        ],
        dim=1,
    )

    # --- Validate target_idx ---
    if target_idx is not None:
        try:
            target_idx = int(target_idx)
        except Exception:
            target_idx = None
        if target_idx is not None and not (0 <= target_idx < S):
            target_idx = None

    # --- AA overlay for center markers ONLY (keeps rectangles unchanged / non-AA) ---
    aa_scale = 4  # you can parameterize this if you want
    aa = int(max(1, aa_scale))
    overlay_hi = None
    draw_hi = None

    # Only create the overlay if we will draw markers at all
    # (marker_alpha is only not None when influence_pct is not None)
    if influence_pct is not None:
        overlay_hi = Image.new("RGBA", (W * aa, H * aa), (0, 0, 0, 0))
        draw_hi = ImageDraw.Draw(overlay_hi, "RGBA")

    k = 10

    # --- Draw elements ---
    for i in range(S):
        cat_raw = feats_t[i].item()
        cat = int(cat_raw) - 1
        if cat < 0:
            continue

        col = colors[cat] if 0 <= cat < len(colors) else [0, 0, 0]

        x1, y1, x2, y2 = box[i].tolist()
        outline_width = DEFAULT_WIDTH
        marker_alpha = 255
        fill_alpha = 64
        outline_alpha = 200
        badge_alpha = 255

        marker_color = (0, 0, 0)
        fill_color = tuple(col)
        outline_color = tuple(col)
        badge_color = (0, 0, 0)

        if influence_mode == "grouped_all" or influence_mode == "grouped_psc":
            if influence_mode == "grouped_all":
                p = float(influence_pct[i, 0].item())  # [0,1]
                a = int(round(p * 255))

                fill_alpha = a
                outline_alpha = a
            # "grouped_psc" influence mode
            else:
                s_inf = (influence[i, 1] * k).round()
                outline_width = s_inf

                fill_alpha = 128 if (target_idx is not None and i == target_idx) else 64

        fill_rgba = fill_color + (fill_alpha, )
        outline_rgba = outline_color + (outline_alpha, )

        draw.rectangle([x1, y1, x2, y2], fill=fill_rgba)
        draw.rectangle([x1, y1, x2, y2], outline=outline_rgba, width=int(outline_width))

        if influence_mode == "grouped_psc":
            t_inf = (influence[i, 2] * k).round()

            if t_inf != 0:
                half = t_inf / 2.0

                bx1, by1 = x2 - half, y2 - half
                bx2, by2 = x2 + half, y2 + half

                badge_rgba = badge_color + (badge_alpha,)

                draw.rectangle([bx1, by1, bx2, by2], fill=badge_rgba)

        if (influence_mode == "grouped_all" and i == target_idx) or influence_mode == "grouped_psc":
            marker_rgba = marker_color + (marker_alpha,)

            if influence_mode == "grouped_all":
                marker_width = DEFAULT_WIDTH + MARKER_WIDTH_ADJ
            else:
                p_inf = (influence[i, 0] * k).round()
                marker_width = p_inf

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            r = marker_width / 2

            # draw at supersampled resolution
            cx_s, cy_s = cx * aa, cy * aa
            r_s = r * aa
            bbox_s = [cx_s - r_s, cy_s - r_s, cx_s + r_s, cy_s + r_s]

            draw_hi.ellipse(bbox_s, fill=marker_rgba)

    # --- Composite AA markers back onto base ---
    if overlay_hi is not None:
        overlay = overlay_hi.resize((W, H), resample=Image.LANCZOS)
        img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")

    return img


def visualize_trajectory(
        cfg,
        batch,
        full_geom_pred,
        full_cat_pred,
        instance_idx,
        influence,
        target_idx,
        *,
        num_colors=26,
        square=False,
        influence_mode="grouped_all",
        out_dir=None,
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

    fps = 8.0
    duration_ms = int(round(1000.0 / fps))

    # Render frames
    frames = []
    frames_xai = []
    T = int(traj_bbox_xywh.shape[0])

    max_mag_global = None
    ig_return_xy = influence_mode == "per_xy"

    if ig_return_xy:
        infl_xy = influence[:, 0, :L].detach().cpu()  # [T, L, 2]
        mags = torch.sqrt(infl_xy[..., 0] ** 2 + infl_xy[..., 1] ** 2).reshape(-1)
        if mags.numel():
            max_mag_global = float(torch.quantile(mags, 0.95).item())  # robust scaling
            max_mag_global = max(max_mag_global, 1e-6)
        else:
            max_mag_global = 1.0

    for t in range(T):
        img = draw_xai_layout(
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
                influence=influence[t, 0, :L].detach().cpu(),  # [L,1] or [L,3]
                target_idx=target_idx,
                influence_mode=influence_mode,
                t=t
            )

        # Ensure GIF-compatible mode
        # (RGBA is okay, but many viewers handle palette GIFs better)
        frames.append(img.convert("P", palette=Image.ADAPTIVE))
        frames_xai.append(img_xai.convert("P", palette=Image.ADAPTIVE))

    if len(frames) != 0:
        gif_path = os.path.join(out_dir, f"{instance_idx}.gif")

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

    if len(frames_xai) != 0:
        gif_path = os.path.join(out_dir, f"{instance_idx}_{target_idx}.gif")

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
