import argparse
import csv
import math
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import sklearn.metrics as skm
except ImportError:
    skm = None

try:
    from scipy.signal import convolve2d
except ImportError:
    convolve2d = None

try:
    from skimage.metrics import structural_similarity as skimage_ssim
except ImportError:
    skimage_ssim = None


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def read_gray(path):
    if cv2 is not None:
        image_bgr = cv2.imread(str(path))
        if image_bgr is None:
            raise RuntimeError(f"Failed to read image: {path}")
        return np.round(cv2.cvtColor(image_bgr.astype("float32"), cv2.COLOR_BGR2GRAY)).astype(np.float64)
    return np.asarray(Image.open(path).convert("L"), dtype=np.float64)


def list_pairs(ir_dir, vi_dir, fused_dir):
    ir_files = {p.stem: p for p in Path(ir_dir).iterdir() if p.suffix.lower() in IMG_EXTS}
    vi_files = {p.stem: p for p in Path(vi_dir).iterdir() if p.suffix.lower() in IMG_EXTS}
    fused_files = {p.stem: p for p in Path(fused_dir).iterdir() if p.suffix.lower() in IMG_EXTS}
    keys = sorted(set(ir_files) & set(vi_files) & set(fused_files))
    return [(key, ir_files[key], vi_files[key], fused_files[key]) for key in keys]


def resize_like(image, shape):
    if image.shape == shape:
        return image
    pil = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))
    pil = pil.resize((shape[1], shape[0]), Image.BILINEAR)
    return np.asarray(pil, dtype=np.float64)


def entropy(image):
    a = np.uint8(np.round(np.clip(image, 0, 255))).flatten()
    h = np.bincount(a) / a.shape[0]
    return float(-sum(h * np.log2(h + (h == 0))))


def standard_deviation(image):
    return float(np.std(image))


def spatial_frequency(image):
    return float(np.sqrt(np.mean((image[:, 1:] - image[:, :-1]) ** 2) + np.mean((image[1:, :] - image[:-1, :]) ** 2)))


def average_gradient(image):
    image = image.astype(np.float32)
    if cv2 is not None:
        gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
        return float(np.mean(np.sqrt((gx * gx + gy * gy) / 2.0)))
    dx = np.diff(image, axis=1)
    dy = np.diff(image, axis=0)
    dx = dx[:-1, :]
    dy = dy[:, :-1]
    return float(np.mean(np.sqrt((dx * dx + dy * dy) / 2.0)))


def mutual_information(a, b):
    a = np.uint8(np.round(np.clip(a, 0, 255))).flatten()
    b = np.uint8(np.round(np.clip(b, 0, 255))).flatten()
    if skm is not None:
        return float(skm.mutual_info_score(a, b))

    hist, _, _ = np.histogram2d(a, b, bins=256, range=[[0, 255], [0, 255]])
    pxy = hist / (np.sum(hist) + 1e-12)
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    px_py = px[:, None] * py[None, :]
    nz = pxy > 0
    return float(np.sum(pxy[nz] * np.log(pxy[nz] / (px_py[nz] + 1e-12))))


def correlation(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    a = a - np.mean(a)
    b = b - np.mean(b)
    denom = np.sqrt(np.sum(a * a) * np.sum(b * b)) + 1e-12
    return float(np.sum(a * b) / denom)


def scd(ir, vi, fused):
    img_f_ir = fused - ir
    img_f_vi = fused - vi
    return correlation(ir, img_f_vi) + correlation(vi, img_f_ir)


def box_filter(image, radius):
    if radius <= 0:
        return image.astype(np.float64)
    pad = radius
    image = image.astype(np.float64)
    padded = np.pad(image, ((pad, pad), (pad, pad)), mode="reflect")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
    size = 2 * radius + 1
    total = integral[size:, size:] - integral[:-size, size:] - integral[size:, :-size] + integral[:-size, :-size]
    return total / float(size * size)


def ssim_pair(a, b):
    if skimage_ssim is not None:
        return float(skimage_ssim(a, b, data_range=255))

    a = a.astype(np.float64)
    b = b.astype(np.float64)
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    mu_a = box_filter(a, 5)
    mu_b = box_filter(b, 5)
    sigma_a2 = box_filter(a * a, 5) - mu_a * mu_a
    sigma_b2 = box_filter(b * b, 5) - mu_b * mu_b
    sigma_ab = box_filter(a * b, 5) - mu_a * mu_b
    val = ((2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)) / (
        (mu_a * mu_a + mu_b * mu_b + c1) * (sigma_a2 + sigma_b2 + c2) + 1e-12
    )
    return float(np.mean(val))


def sobel(image):
    if cv2 is not None:
        image = image.astype(np.float32)
        gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3).astype(np.float64)
        gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3).astype(np.float64)
        mag = np.sqrt(gx * gx + gy * gy)
        angle = np.arctan2(gy, gx)
        return mag, angle
    p = np.pad(image.astype(np.float64), ((1, 1), (1, 1)), mode="reflect")
    gx = (
        -p[:-2, :-2]
        - 2 * p[1:-1, :-2]
        - p[2:, :-2]
        + p[:-2, 2:]
        + 2 * p[1:-1, 2:]
        + p[2:, 2:]
    )
    gy = (
        -p[:-2, :-2]
        - 2 * p[:-2, 1:-1]
        - p[:-2, 2:]
        + p[2:, :-2]
        + 2 * p[2:, 1:-1]
        + p[2:, 2:]
    )
    mag = np.sqrt(gx * gx + gy * gy)
    angle = np.arctan2(gy, gx)
    return mag, angle


def conv2_same(image, kernel):
    if convolve2d is not None:
        return convolve2d(image, kernel, mode="same")
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(image, ((ph, ph), (pw, pw)), mode="constant")
    out = np.zeros_like(image, dtype=np.float64)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            out[y, x] = np.sum(padded[y : y + kh, x : x + kw] * kernel)
    return out


def qabf_get_array(image):
    h1 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
    h3 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sx = conv2_same(image, h3)
    sy = conv2_same(image, h1)
    g = np.sqrt(np.multiply(sx, sx) + np.multiply(sy, sy))
    a = np.zeros_like(image)
    a[sx == 0] = math.pi / 2
    a[sx != 0] = np.arctan(sy[sx != 0] / sx[sx != 0])
    return g, a


def qabf_get_qabf(a_src, g_src, a_fused, g_fused):
    tg = 0.9994
    kg = -15
    dg = 0.5
    ta = 0.9879
    ka = -22
    da = 0.8
    gaf = np.zeros_like(a_src)
    gaf[g_src > g_fused] = g_fused[g_src > g_fused] / g_src[g_src > g_fused]
    gaf[g_src == g_fused] = g_fused[g_src == g_fused]
    gaf[g_src < g_fused] = g_src[g_src < g_fused] / g_fused[g_src < g_fused]
    aaf = 1 - np.abs(a_src - a_fused) / (math.pi / 2)
    qgaf = tg / (1 + np.exp(kg * (gaf - dg)))
    qaaf = ta / (1 + np.exp(ka * (aaf - da)))
    return qgaf * qaaf


def edge_transfer_one_source(src, fused):
    g_src, a_src = sobel(src)
    g_fused, a_fused = sobel(fused)
    g_ratio = np.minimum(g_src, g_fused) / (np.maximum(g_src, g_fused) + 1e-12)
    angle_diff = np.abs(a_src - a_fused)
    angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)
    angle_ratio = 1.0 - angle_diff / (np.pi / 2.0)
    angle_ratio = np.clip(angle_ratio, 0.0, 1.0)

    tg, kg, dg = 0.9994, -15.0, 0.5
    ta, ka, da = 0.9879, -22.0, 0.8
    q_g = tg / (1.0 + np.exp(kg * (g_ratio - dg)))
    q_a = ta / (1.0 + np.exp(ka * (angle_ratio - da)))
    q = q_g * q_a
    false_edge = (g_fused > g_src).astype(np.float64) * (1.0 - q)
    return q, false_edge, g_src


def qabf(ir, vi, fused):
    g_ir, a_ir = qabf_get_array(ir)
    g_vi, a_vi = qabf_get_array(vi)
    g_fused, a_fused = qabf_get_array(fused)
    q_ir = qabf_get_qabf(a_ir, g_ir, a_fused, g_fused)
    q_vi = qabf_get_qabf(a_vi, g_vi, a_fused, g_fused)
    return float(np.sum(q_ir * g_ir + q_vi * g_vi) / (np.sum(g_ir + g_vi) + 1e-12))


def nabf(ir, vi, fused):
    # Same calculation style as the reference code:
    # false_edge = max(gradient(fused) - max(gradient(ir), gradient(vi)), 0)
    # Nabf = sum(false_edge) / sum(gradient(fused)).
    gf, _ = sobel(fused)
    gir, _ = sobel(ir)
    gvi, _ = sobel(vi)
    gsrc = np.maximum(gir, gvi)
    false_edge = np.maximum(gf - gsrc, 0.0)
    return float(np.sum(false_edge) / (np.sum(gf) + 1e-12))


def vif_single(ref, dist):
    # Same VIFF implementation as the reference Evaluator.compare_viff.
    if convolve2d is None:
        return vif_single_box_fallback(ref, dist)
    sigma_nsq = 2.0
    eps = 1e-10
    ref = ref.astype(np.float64)
    dist = dist.astype(np.float64)
    num = 0.0
    den = 0.0
    for scale in range(1, 5):
        n = 2 ** (4 - scale + 1) + 1
        sd = n / 5.0
        m, n2 = [(ss - 1.0) / 2.0 for ss in (n, n)]
        y, x = np.ogrid[-m : m + 1, -n2 : n2 + 1]
        h = np.exp(-(x * x + y * y) / (2.0 * sd * sd))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        sumh = h.sum()
        win = h / sumh if sumh != 0 else h

        if scale > 1:
            ref = convolve2d(ref, np.rot90(win, 2), mode="valid")
            dist = convolve2d(dist, np.rot90(win, 2), mode="valid")
            ref = ref[::2, ::2]
            dist = dist[::2, ::2]

        mu_ref = convolve2d(ref, np.rot90(win, 2), mode="valid")
        mu_dist = convolve2d(dist, np.rot90(win, 2), mode="valid")
        sigma_ref_sq = convolve2d(ref * ref, np.rot90(win, 2), mode="valid") - mu_ref * mu_ref
        sigma_dist_sq = convolve2d(dist * dist, np.rot90(win, 2), mode="valid") - mu_dist * mu_dist
        sigma_ref_dist = convolve2d(ref * dist, np.rot90(win, 2), mode="valid") - mu_ref * mu_dist
        sigma_ref_sq[sigma_ref_sq < 0] = 0
        sigma_dist_sq[sigma_dist_sq < 0] = 0

        g = sigma_ref_dist / (sigma_ref_sq + eps)
        sv_sq = sigma_dist_sq - g * sigma_ref_dist
        g[sigma_ref_sq < eps] = 0
        sv_sq[sigma_ref_sq < eps] = sigma_dist_sq[sigma_ref_sq < eps]
        sigma_ref_sq[sigma_ref_sq < eps] = 0
        g[sigma_dist_sq < eps] = 0
        sv_sq[sigma_dist_sq < eps] = 0
        sv_sq[g < 0] = sigma_dist_sq[g < 0]
        g[g < 0] = 0
        sv_sq[sv_sq <= eps] = eps
        num += np.sum(np.log10(1.0 + (g * g) * sigma_ref_sq / (sv_sq + sigma_nsq)))
        den += np.sum(np.log10(1.0 + sigma_ref_sq / sigma_nsq))
    value = num / den
    return 1.0 if np.isnan(value) else float(value)


def vif_single_box_fallback(ref, dist):
    sigma_nsq = 2.0
    ref = ref.astype(np.float64)
    dist = dist.astype(np.float64)
    num = 0.0
    den = 0.0
    for scale in range(4):
        radius = 2 ** scale
        if scale > 0:
            ref = box_filter(ref, 1)[::2, ::2]
            dist = box_filter(dist, 1)[::2, ::2]
        mu_ref = box_filter(ref, radius)
        mu_dist = box_filter(dist, radius)
        sigma_ref_sq = np.maximum(box_filter(ref * ref, radius) - mu_ref * mu_ref, 0.0)
        sigma_dist_sq = np.maximum(box_filter(dist * dist, radius) - mu_dist * mu_dist, 0.0)
        sigma_ref_dist = box_filter(ref * dist, radius) - mu_ref * mu_dist
        g = sigma_ref_dist / (sigma_ref_sq + 1e-10)
        sv_sq = sigma_dist_sq - g * sigma_ref_dist
        g = np.where(sigma_ref_sq < 1e-10, 0.0, g)
        sv_sq = np.where(sigma_ref_sq < 1e-10, sigma_dist_sq, sv_sq)
        sigma_ref_sq = np.where(sigma_ref_sq < 1e-10, 0.0, sigma_ref_sq)
        g = np.maximum(g, 0.0)
        sv_sq = np.maximum(sv_sq, 1e-10)
        num += np.sum(np.log10(1.0 + (g * g) * sigma_ref_sq / (sv_sq + sigma_nsq)))
        den += np.sum(np.log10(1.0 + sigma_ref_sq / sigma_nsq))
    value = num / (den + 1e-12)
    return 1.0 if np.isnan(value) else float(value)


def vif(ir, vi, fused):
    return vif_single(ir, fused) + vif_single(vi, fused)


def metrics_for_pair(ir, vi, fused):
    mi_ir = mutual_information(fused, ir)
    mi_vi = mutual_information(fused, vi)
    ssim_ir = ssim_pair(fused, ir)
    ssim_vi = ssim_pair(fused, vi)
    return {
        "EN": entropy(fused),
        "SD": standard_deviation(fused),
        "SF": spatial_frequency(fused),
        "MI": mi_ir + mi_vi,
        "SCD": scd(ir, vi, fused),
        "VIF": vif(ir, vi, fused),
        "Qabf": qabf(ir, vi, fused),
        "SSIM": ssim_ir + ssim_vi,
        "AG": average_gradient(fused),
        "Nabf": nabf(ir, vi, fused),
    }


def round_metric(value):
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.4f}"
    return value


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: round_metric(value) for key, value in row.items()})


def mean_row(dataset, rows):
    metric_keys = [k for k in rows[0] if k not in {"dataset", "image_id", "ir_path", "vi_path", "fused_path"}]
    row = {"dataset": dataset, "image_id": "MEAN", "ir_path": "", "vi_path": "", "fused_path": ""}
    for key in metric_keys:
        row[key] = float(np.mean([float(r[key]) for r in rows]))
    return row


def evaluate_dataset(root, fused_root, dataset, fused_subdir):
    ir_dir = root / "test" / dataset / "Inf"
    vi_dir = root / "test" / dataset / "Vis"
    fused_dir = fused_root / dataset / fused_subdir
    if not fused_dir.exists() and fused_subdir == "fused_gray":
        fused_dir = fused_root / dataset / "fused_color"
    if not ir_dir.exists() or not vi_dir.exists() or not fused_dir.exists():
        raise FileNotFoundError(f"Missing directories for {dataset}: {ir_dir}, {vi_dir}, {fused_dir}")

    pairs = list_pairs(ir_dir, vi_dir, fused_dir)
    if not pairs:
        raise RuntimeError(f"No paired files found for {dataset}.")

    rows = []
    for idx, (key, ir_path, vi_path, fused_path) in enumerate(pairs, start=1):
        print(f"[{dataset} {idx}/{len(pairs)}] {key}")
        ir = read_gray(ir_path)
        vi = resize_like(read_gray(vi_path), ir.shape)
        fused = resize_like(read_gray(fused_path), ir.shape)
        row = {
            "dataset": dataset,
            "image_id": key,
            "ir_path": str(ir_path),
            "vi_path": str(vi_path),
            "fused_path": str(fused_path),
        }
        row.update(metrics_for_pair(ir, vi, fused))
        rows.append(row)
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate infrared-visible fusion metrics.")
    parser.add_argument("--root", default=".", help="Project root.")
    parser.add_argument("--fused-root", default="runs/regfuse_v2_infer", help="Inference output root.")
    parser.add_argument("--datasets", default="LLVIP,M3FD,MSRS,RoadScene,TNO")
    parser.add_argument("--fused-subdir", default="fused_gray", help="Use fused_gray or fused_color.")
    parser.add_argument("--out", default="metrics/regfuse_v2", help="Metric output directory.")
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.root)
    fused_root = root / args.fused_root
    out_dir = root / args.out
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    summary_rows = []
    for dataset in datasets:
        rows = evaluate_dataset(root, fused_root, dataset, args.fused_subdir)
        dataset_mean = mean_row(dataset, rows)
        write_csv(out_dir / f"{dataset}_metrics.csv", rows + [dataset_mean])
        summary_rows.append(dataset_mean)

    overall = mean_row("ALL", summary_rows)
    write_csv(out_dir / "summary_metrics.csv", summary_rows + [overall])
    print(f"Saved metric CSV files to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
