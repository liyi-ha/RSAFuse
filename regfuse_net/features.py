import math
from pathlib import Path

import numpy as np
from PIL import Image


CLASS_COLORS = np.array(
    [
        [35, 35, 35],
        [230, 55, 45],
        [35, 125, 230],
        [250, 205, 45],
        [70, 190, 110],
    ],
    dtype=np.uint8,
)


def read_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def save_gray(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def save_label(path, label):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(label.astype(np.uint8)).save(path)


def save_rgb(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def rgb_to_y(rgb):
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def ycbcr_chroma(rgb):
    y = rgb_to_y(rgb)
    cb = (rgb[..., 2] - y) * 0.564 + 0.5
    cr = (rgb[..., 0] - y) * 0.713 + 0.5
    return y, cb, cr


def ycbcr_to_rgb(y, cb, cr):
    r = y + 1.403 * (cr - 0.5)
    b = y + 1.773 * (cb - 0.5)
    g = (y - 0.299 * r - 0.114 * b) / 0.587
    return np.clip(np.stack([r, g, b], axis=-1), 0.0, 1.0)


def percentile_normalize(x, low=1.0, high=99.0):
    p_low, p_high = np.percentile(x, [low, high])
    if p_high - p_low < 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - p_low) / (p_high - p_low), 0.0, 1.0).astype(np.float32)


def contrast_stretch(x, low=0.5, high=99.5):
    p_low, p_high = np.percentile(x, [low, high])
    if p_high - p_low < 1e-6:
        return np.clip(x, 0.0, 1.0).astype(np.float32)
    return np.clip((x - p_low) / (p_high - p_low), 0.0, 1.0).astype(np.float32)


def soft_percentile_map(x, low=72.0, high=98.0):
    p_low, p_high = np.percentile(x, [low, high])
    if p_high - p_low < 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - p_low) / (p_high - p_low), 0.0, 1.0).astype(np.float32)


def minmax_normalize(x):
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    if x_max - x_min < 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - x_min) / (x_max - x_min)).astype(np.float32)


def box_blur(image, radius):
    if radius <= 0:
        return image.astype(np.float32)
    pad = radius
    padded = np.pad(image, ((pad, pad), (pad, pad)), mode="reflect")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
    size = 2 * radius + 1
    total = (
        integral[size:, size:]
        - integral[:-size, size:]
        - integral[size:, :-size]
        + integral[:-size, :-size]
    )
    return (total / float(size * size)).astype(np.float32)


def smooth_map(image, radius=7, passes=2):
    out = image.astype(np.float32)
    for _ in range(passes):
        out = box_blur(out, radius)
    return out


def gradient_magnitude(image):
    padded = np.pad(image, ((1, 1), (1, 1)), mode="reflect")
    gx = (
        -padded[:-2, :-2]
        - 2 * padded[1:-1, :-2]
        - padded[2:, :-2]
        + padded[:-2, 2:]
        + 2 * padded[1:-1, 2:]
        + padded[2:, 2:]
    )
    gy = (
        -padded[:-2, :-2]
        - 2 * padded[:-2, 1:-1]
        - padded[:-2, 2:]
        + padded[2:, :-2]
        + 2 * padded[2:, 1:-1]
        + padded[2:, 2:]
    )
    return np.sqrt(gx * gx + gy * gy).astype(np.float32)


def local_texture(image, radius=3):
    mean = box_blur(image, radius)
    mean_sq = box_blur(image * image, radius)
    var = np.maximum(mean_sq - mean * mean, 0.0)
    return np.sqrt(var).astype(np.float32)


def compute_features_np(infrared_y, visible_y):
    i_norm = percentile_normalize(infrared_y)
    v_norm = percentile_normalize(visible_y)

    grad_i = minmax_normalize(gradient_magnitude(i_norm))
    grad_v = minmax_normalize(gradient_magnitude(v_norm))
    tex_i = minmax_normalize(local_texture(i_norm, radius=4))
    tex_v = minmax_normalize(local_texture(v_norm, radius=4))

    d_l = np.abs(i_norm - v_norm)
    d_g = np.abs(grad_i - grad_v)
    d_t = np.abs(tex_i - tex_v)

    hot_contrast = np.maximum(i_norm - box_blur(i_norm, radius=13), 0.0)
    hot_i = minmax_normalize(hot_contrast)
    thermal_saliency = minmax_normalize(0.70 * hot_i + 0.20 * i_norm + 0.10 * grad_i)
    visible_detail = minmax_normalize(0.65 * grad_v + 0.35 * tex_v)

    return {
        "I": i_norm,
        "V": v_norm,
        "grad_i": grad_i,
        "grad_v": grad_v,
        "tex_i": tex_i,
        "tex_v": tex_v,
        "D_l": d_l,
        "D_g": d_g,
        "D_t": d_t,
        "hot_i": hot_i,
        "S_i": thermal_saliency,
        "S_v": visible_detail,
    }


def feature_stack_np(infrared_rgb, visible_rgb):
    infrared_y = rgb_to_y(infrared_rgb)
    visible_y = rgb_to_y(visible_rgb)
    f = compute_features_np(infrared_y, visible_y)
    stack = np.stack(
        [f["I"], f["V"], f["D_l"], f["D_g"], f["D_t"], f["hot_i"], f["S_i"], f["S_v"]],
        axis=0,
    )
    return stack.astype(np.float32), f


def block_means(feature, block_size):
    h, w = feature.shape
    bh = int(math.ceil(h / block_size))
    bw = int(math.ceil(w / block_size))
    out = np.zeros((bh, bw), dtype=np.float32)
    for by in range(bh):
        y0 = by * block_size
        y1 = min((by + 1) * block_size, h)
        for bx in range(bw):
            x0 = bx * block_size
            x1 = min((bx + 1) * block_size, w)
            out[by, bx] = float(np.mean(feature[y0:y1, x0:x1]))
    return out


def expand_blocks(blocks, shape, block_size):
    h, w = shape
    out = np.zeros((h, w), dtype=blocks.dtype)
    for by in range(blocks.shape[0]):
        y0 = by * block_size
        y1 = min((by + 1) * block_size, h)
        for bx in range(blocks.shape[1]):
            x0 = bx * block_size
            x1 = min((bx + 1) * block_size, w)
            out[y0:y1, x0:x1] = blocks[by, bx]
    return out


def classify_difference_regions(features, block_size=24):
    h, w = features["I"].shape
    names = ["D_l", "D_g", "D_t", "hot_i", "S_i", "S_v", "grad_i", "grad_v", "tex_i", "tex_v"]
    bm = {name: block_means(features[name], block_size) for name in names}

    thresholds = {}
    for name, values in bm.items():
        thresholds[name] = {
            "low": float(np.percentile(values, 35)),
            "mid": float(np.percentile(values, 50)),
            "high": float(np.percentile(values, 70)),
        }

    blocks = np.full_like(bm["D_l"], 4, dtype=np.uint8)
    for by in range(blocks.shape[0]):
        for bx in range(blocks.shape[1]):
            dl = bm["D_l"][by, bx]
            dg = bm["D_g"][by, bx]
            dt = bm["D_t"][by, bx]
            hot = bm["hot_i"][by, bx]
            si = bm["S_i"][by, bx]
            sv = bm["S_v"][by, bx]
            gi = bm["grad_i"][by, bx]
            gv = bm["grad_v"][by, bx]

            if dl < thresholds["D_l"]["low"] and dg < thresholds["D_g"]["low"] and dt < thresholds["D_t"]["low"]:
                cls = 0
            elif hot > thresholds["hot_i"]["mid"] and si > thresholds["S_i"]["high"] and si > 1.10 * sv:
                cls = 1
            elif sv > thresholds["S_v"]["high"] and (
                sv > si or gv > 1.05 * gi or dg > thresholds["D_g"]["mid"] or dt > thresholds["D_t"]["mid"]
            ):
                cls = 2
            elif dl > thresholds["D_l"]["high"] and dg < thresholds["D_g"]["mid"] and dt < thresholds["D_t"]["mid"]:
                cls = 3
            else:
                cls = 4
            blocks[by, bx] = cls

    return expand_blocks(blocks, (h, w), block_size)


def build_visual_alpha(features, mask, smooth_radius=2, thermal_strength=0.68, background_ir=0.035):
    infrared_advantage = np.maximum(features["I"] - features["V"], 0.0)
    thermal_score = minmax_normalize(
        0.58 * features["hot_i"] + 0.24 * features["S_i"] + 0.18 * infrared_advantage
    )
    thermal_target = soft_percentile_map(thermal_score, low=68.0, high=97.8)
    thermal_target = np.maximum(thermal_target, 0.65 * soft_percentile_map(features["hot_i"], low=72.0, high=98.5))

    alpha = background_ir + thermal_strength * thermal_target
    alpha = np.where(mask == 2, np.minimum(alpha, 0.10 + 0.45 * thermal_target), alpha)
    alpha = np.where(mask == 0, np.minimum(alpha, 0.12 + 0.50 * thermal_target), alpha)
    alpha = np.where(mask == 1, np.maximum(alpha, background_ir + 0.64 * thermal_target), alpha)
    alpha = smooth_map(alpha.astype(np.float32), radius=smooth_radius, passes=1)
    thermal_target = smooth_map(thermal_target.astype(np.float32), radius=1, passes=1)
    return np.clip(alpha, 0.02, 0.76), np.clip(thermal_target, 0.0, 1.0)


def generate_teacher_np(infrared_rgb, visible_rgb, block_size=24, smooth_radius=7):
    _, features = feature_stack_np(infrared_rgb, visible_rgb)
    mask = classify_difference_regions(features, block_size=block_size)
    alpha, thermal_target = build_visual_alpha(features, mask, smooth_radius=max(1, smooth_radius // 3))
    return mask.astype(np.uint8), alpha.astype(np.float32), thermal_target.astype(np.float32), features


def colorize_mask(mask):
    return CLASS_COLORS[mask.astype(np.uint8)]


def thermal_desaturate_chroma(cb, cr, thermal_target, strength=0.78):
    desat = np.clip(strength * thermal_target, 0.0, 0.92)
    cb_out = (1.0 - desat) * cb + desat * 0.5
    cr_out = (1.0 - desat) * cr + desat * 0.5
    return cb_out, cr_out


def fuse_visible_guided_np(infrared_rgb, visible_rgb, alpha, detail_gain=0.62, sharpen_amount=0.26):
    stack, features = feature_stack_np(infrared_rgb, visible_rgb)
    visible_y, cb, cr = ycbcr_chroma(visible_rgb)
    visible = features["V"]
    infrared = features["I"]

    visible_tone = contrast_stretch(visible, low=0.25, high=99.85)
    visible_tone = np.clip(np.power(visible_tone, 0.88), 0.0, 1.0)
    visible_detail = visible_tone - smooth_map(visible_tone, radius=2, passes=1)
    visible_tone = np.clip(visible_tone + 0.36 * visible_detail, 0.0, 1.0)

    infrared_tone = contrast_stretch(infrared, low=0.3, high=99.7)
    infrared_tone = np.clip(0.10 + 0.82 * infrared_tone, 0.0, 0.92)

    alpha = np.clip(alpha.astype(np.float32), 0.0, 1.0)
    fused_y = (1.0 - alpha) * visible_tone + alpha * infrared_tone

    detail_i = infrared - smooth_map(infrared, radius=2, passes=1)
    detail_v = visible - smooth_map(visible, radius=2, passes=1)
    detail = alpha * (0.55 * detail_i + 0.45 * detail_v) + (1.0 - alpha) * detail_v
    fused_y = np.clip(fused_y + detail_gain * detail, 0.0, 1.0)

    blur_small = smooth_map(fused_y, radius=1, passes=1)
    fused_y = np.clip(fused_y + sharpen_amount * (fused_y - blur_small), 0.0, 1.0)
    fused_y = contrast_stretch(fused_y, low=0.15, high=99.9)

    cb_fused, cr_fused = thermal_desaturate_chroma(cb, cr, alpha)
    fused_rgb = ycbcr_to_rgb(fused_y, cb_fused, cr_fused)
    return fused_y.astype(np.float32), fused_rgb.astype(np.float32)
