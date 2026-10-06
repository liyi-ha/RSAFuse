import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from regfuse_net.dataset import list_paired_files
from regfuse_net.features import colorize_mask, feature_stack_np, read_rgb, save_gray, save_rgb, ycbcr_chroma, ycbcr_to_rgb
from regfuse_net.losses import compose_fusion
from regfuse_net.model import RegFuseUNet


def parse_args():
    parser = argparse.ArgumentParser(description="Run RegFuse network inference.")
    parser.add_argument("--root", default=".", help="Project root.")
    parser.add_argument("--checkpoint", default="checkpoints/regfuse_v2/best.pth")
    parser.add_argument("--out", default="runs/regfuse_v2_infer")
    parser.add_argument("--base-channels", type=int, default=0, help="0 means read from checkpoint args.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--datasets", default="LLVIP,M3FD,MSRS,RoadScene")
    parser.add_argument("--detail-gain", type=float, default=0.62)
    parser.add_argument("--sharpen", type=float, default=0.26)
    return parser.parse_args()


def resize_like(visible_rgb, target_shape):
    if visible_rgb.shape[:2] == target_shape:
        return visible_rgb
    img = Image.fromarray(np.clip(visible_rgb * 255.0, 0, 255).astype(np.uint8))
    img = img.resize((target_shape[1], target_shape[0]), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def load_model(args):
    ckpt = torch.load(Path(args.root) / args.checkpoint, map_location=args.device)
    ckpt_args = ckpt.get("args", {})
    base_channels = args.base_channels or int(ckpt_args.get("base_channels", 32))
    model = RegFuseUNet(in_channels=8, num_classes=5, base_channels=base_channels).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}, base_channels={base_channels}")
    return model


@torch.no_grad()
def infer_pair(model, ir_path, vi_path, args):
    ir = read_rgb(ir_path)
    vi = resize_like(read_rgb(vi_path), ir.shape[:2])
    x, _ = feature_stack_np(ir, vi)
    x_t = torch.from_numpy(x[None, ...]).float().to(args.device)
    seg_logits, alpha_pred, detail_gate, residual_pred = model(x_t)
    seg_prob = torch.softmax(seg_logits, dim=1)
    mask = torch.argmax(seg_prob, dim=1)[0].cpu().numpy().astype(np.uint8)
    fused_t = compose_fusion(
        x_t,
        x_t[:, 0:1],
        x_t[:, 1:2],
        alpha_pred,
        detail_gate,
        residual_pred,
        residual_scale=0.10,
        detail_scale=args.detail_gain,
    )
    alpha = alpha_pred[0, 0].cpu().numpy().astype(np.float32)
    fused_y = fused_t[0, 0].cpu().numpy().astype(np.float32)
    _, cb, cr = ycbcr_chroma(vi)
    fused_rgb = ycbcr_to_rgb(fused_y, cb, cr)
    return fused_y, fused_rgb, alpha, mask


def main():
    args = parse_args()
    root = Path(args.root)
    out_root = root / args.out
    model = load_model(args)
    dataset_names = [name.strip() for name in args.datasets.split(",") if name.strip()]

    for dataset in dataset_names:
        ir_dir = root / "test" / dataset / "Inf"
        vi_dir = root / "test" / dataset / "Vis"
        if not ir_dir.exists() or not vi_dir.exists():
            print(f"Skip {dataset}: missing {ir_dir} or {vi_dir}")
            continue
        pairs = list_paired_files(ir_dir, vi_dir)
        print(f"{dataset}: {len(pairs)} pairs")
        for idx, (key, ir_path, vi_path) in enumerate(pairs, start=1):
            print(f"[{dataset} {idx}/{len(pairs)}] {key}")
            fused_y, fused_rgb, alpha, mask = infer_pair(model, ir_path, vi_path, args)
            base = out_root / dataset
            save_rgb(base / "fused_color" / f"{key}.png", fused_rgb)
            save_gray(base / "fused_gray" / f"{key}.png", fused_y)
            save_gray(base / "alpha" / f"{key}.png", alpha)
            save_rgb(base / "mask_color" / f"{key}.png", colorize_mask(mask).astype(np.float32) / 255.0)
    print(f"Inference results saved to: {out_root.resolve()}")


if __name__ == "__main__":
    main()
