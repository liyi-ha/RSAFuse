import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .features import feature_stack_np, fuse_visible_guided_np, generate_teacher_np, read_rgb


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_paired_files(ir_dir, vi_dir, split_file=None):
    ir_dir = Path(ir_dir)
    vi_dir = Path(vi_dir)
    ir_files = {p.stem: p for p in ir_dir.iterdir() if p.suffix.lower() in IMG_EXTS}
    vi_files = {p.stem: p for p in vi_dir.iterdir() if p.suffix.lower() in IMG_EXTS}

    if split_file is not None:
        keys = [line.strip() for line in Path(split_file).read_text(encoding="utf-8").splitlines() if line.strip()]
        keys = [key for key in keys if key in ir_files and key in vi_files]
    else:
        keys = sorted(set(ir_files) & set(vi_files))
    return [(key, ir_files[key], vi_files[key]) for key in keys]


def load_label(path):
    return np.asarray(Image.open(path), dtype=np.uint8)


def load_alpha(path):
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def random_crop_params(h, w, crop_size):
    if crop_size <= 0 or (h <= crop_size and w <= crop_size):
        return 0, 0, h, w
    crop_h = min(crop_size, h)
    crop_w = min(crop_size, w)
    y = random.randint(0, h - crop_h)
    x = random.randint(0, w - crop_w)
    return y, x, crop_h, crop_w


def center_crop_params(h, w, crop_size):
    if crop_size <= 0 or (h <= crop_size and w <= crop_size):
        return 0, 0, h, w
    crop_h = min(crop_size, h)
    crop_w = min(crop_size, w)
    y = max((h - crop_h) // 2, 0)
    x = max((w - crop_w) // 2, 0)
    return y, x, crop_h, crop_w


def crop_array(arr, params):
    y, x, h, w = params
    return arr[y : y + h, x : x + w, ...] if arr.ndim == 3 else arr[y : y + h, x : x + w]


class RegFuseDataset(Dataset):
    def __init__(
        self,
        ir_dir,
        vi_dir,
        split_file=None,
        pseudo_root=None,
        crop_size=256,
        train=True,
        on_the_fly_pseudo=False,
        block_size=24,
        smooth_radius=7,
    ):
        self.pairs = list_paired_files(ir_dir, vi_dir, split_file)
        if not self.pairs:
            raise RuntimeError("No paired images found.")
        self.crop_size = int(crop_size)
        self.train = train
        self.on_the_fly_pseudo = on_the_fly_pseudo
        self.block_size = int(block_size)
        self.smooth_radius = int(smooth_radius)
        self.pseudo_root = Path(pseudo_root) if pseudo_root else None

    def __len__(self):
        return len(self.pairs)

    def _load_pseudo(self, key):
        if self.pseudo_root is None:
            return None, None
        mask_path = self.pseudo_root / "mask_cls" / (key + ".png")
        alpha_path = self.pseudo_root / "alpha" / (key + ".png")
        if not mask_path.exists() or not alpha_path.exists():
            return None, None
        return load_label(mask_path), load_alpha(alpha_path)

    def __getitem__(self, index):
        key, ir_path, vi_path = self.pairs[index]
        ir = read_rgb(ir_path)
        vi = read_rgb(vi_path)
        if ir.shape[:2] != vi.shape[:2]:
            vi_img = Image.fromarray(np.clip(vi * 255.0, 0, 255).astype(np.uint8))
            vi_img = vi_img.resize((ir.shape[1], ir.shape[0]), Image.BILINEAR)
            vi = np.asarray(vi_img, dtype=np.float32) / 255.0

        mask, alpha = self._load_pseudo(key)
        if mask is None or alpha is None:
            if not self.on_the_fly_pseudo:
                raise RuntimeError("Missing pseudo labels. Run prepare_pseudo_dataset.py or use --on-the-fly-pseudo.")
            mask, alpha, _, _ = generate_teacher_np(ir, vi, block_size=self.block_size, smooth_radius=self.smooth_radius)

        h, w = ir.shape[:2]
        crop_params = random_crop_params(h, w, self.crop_size) if self.train else center_crop_params(h, w, self.crop_size)
        ir = crop_array(ir, crop_params)
        vi = crop_array(vi, crop_params)
        mask = crop_array(mask, crop_params)
        alpha = crop_array(alpha, crop_params)

        if self.train:
            if random.random() < 0.5:
                ir = np.flip(ir, axis=1).copy()
                vi = np.flip(vi, axis=1).copy()
                mask = np.flip(mask, axis=1).copy()
                alpha = np.flip(alpha, axis=1).copy()
            if random.random() < 0.5:
                ir = np.flip(ir, axis=0).copy()
                vi = np.flip(vi, axis=0).copy()
                mask = np.flip(mask, axis=0).copy()
                alpha = np.flip(alpha, axis=0).copy()

        x, _ = feature_stack_np(ir, vi)
        ir_y = x[0:1]
        vi_y = x[1:2]
        fused_teacher, _ = fuse_visible_guided_np(ir, vi, alpha)

        return {
            "key": key,
            "x": torch.from_numpy(x).float(),
            "ir": torch.from_numpy(ir_y).float(),
            "vi": torch.from_numpy(vi_y).float(),
            "mask": torch.from_numpy(mask.astype(np.int64)).long(),
            "alpha": torch.from_numpy(alpha[None, ...].astype(np.float32)).float(),
            "fused_teacher": torch.from_numpy(fused_teacher[None, ...].astype(np.float32)).float(),
        }
