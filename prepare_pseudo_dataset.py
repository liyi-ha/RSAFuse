import argparse
import random
from pathlib import Path

from regfuse_net.dataset import list_paired_files
from regfuse_net.features import colorize_mask, generate_teacher_np, read_rgb, save_gray, save_label, save_rgb


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare pseudo mask and alpha labels for RegFuse training.")
    parser.add_argument("--root", default=".", help="Project root.")
    parser.add_argument("--ir-dir", default="MSRS_train/ir", help="Training infrared directory.")
    parser.add_argument("--vi-dir", default="MSRS_train/vi", help="Training visible directory.")
    parser.add_argument("--out", default="pseudo_labels/MSRS_train", help="Pseudo label output directory.")
    parser.add_argument("--splits", default="splits", help="Split output directory.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation ratio.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--block-size", type=int, default=24)
    parser.add_argument("--smooth-radius", type=int, default=7)
    parser.add_argument("--limit", type=int, default=0, help="Only process first N pairs, 0 means all.")
    return parser.parse_args()


def write_split(path, keys):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(keys) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    root = Path(args.root)
    ir_dir = root / args.ir_dir
    vi_dir = root / args.vi_dir
    out = root / args.out
    split_dir = root / args.splits
    mask_dir = out / "mask_cls"
    mask_color_dir = out / "mask_color"
    alpha_dir = out / "alpha"
    mask_dir.mkdir(parents=True, exist_ok=True)
    mask_color_dir.mkdir(parents=True, exist_ok=True)
    alpha_dir.mkdir(parents=True, exist_ok=True)

    pairs = list_paired_files(ir_dir, vi_dir)
    if args.limit > 0:
        pairs = pairs[: args.limit]
    if not pairs:
        raise RuntimeError("No paired training images found.")

    keys = []
    for idx, (key, ir_path, vi_path) in enumerate(pairs, start=1):
        print(f"[{idx}/{len(pairs)}] pseudo labels for {key}")
        ir = read_rgb(ir_path)
        vi = read_rgb(vi_path)
        mask, alpha, _, _ = generate_teacher_np(
            ir,
            vi,
            block_size=args.block_size,
            smooth_radius=args.smooth_radius,
        )
        save_label(mask_dir / f"{key}.png", mask)
        save_rgb(mask_color_dir / f"{key}.png", colorize_mask(mask).astype("float32") / 255.0)
        save_gray(alpha_dir / f"{key}.png", alpha)
        keys.append(key)

    rng = random.Random(args.seed)
    rng.shuffle(keys)
    val_count = max(1, int(round(len(keys) * args.val_ratio)))
    val_keys = sorted(keys[:val_count])
    train_keys = sorted(keys[val_count:])
    all_keys = sorted(keys)
    write_split(split_dir / "msrs_all.txt", all_keys)
    write_split(split_dir / "msrs_train.txt", train_keys)
    write_split(split_dir / "msrs_val.txt", val_keys)
    print(f"Pseudo labels saved to: {out.resolve()}")
    print(f"Train/val split: {len(train_keys)} train, {len(val_keys)} val")


if __name__ == "__main__":
    main()
