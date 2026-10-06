import argparse
import json
from pathlib import Path

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from regfuse_net.dataset import RegFuseDataset
from regfuse_net.losses import total_loss
from regfuse_net.model import RegFuseUNet, count_parameters


def parse_args():
    parser = argparse.ArgumentParser(description="Train RegFuse difference-semantic fusion network.")
    parser.add_argument("--root", default=".", help="Project root.")
    parser.add_argument("--ir-dir", default="MSRS_train/ir")
    parser.add_argument("--vi-dir", default="MSRS_train/vi")
    parser.add_argument("--pseudo-root", default="pseudo_labels/MSRS_train")
    parser.add_argument("--train-split", default="splits/msrs_train.txt")
    parser.add_argument("--val-split", default="splits/msrs_val.txt")
    parser.add_argument("--save-dir", default="checkpoints/regfuse_v2")
    parser.add_argument("--resume", default="", help="Resume from a V2 checkpoint.")
    parser.add_argument("--keep-best", action="store_true", help="Keep best_val from resumed checkpoint.")
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true", help="Use mixed precision.")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--on-the-fly-pseudo", action="store_true", help="Generate pseudo labels inside dataset.")
    parser.add_argument("--block-size", type=int, default=24)
    parser.add_argument("--smooth-radius", type=int, default=7)
    parser.add_argument("--w-seg", type=float, default=1.0)
    parser.add_argument("--w-alpha", type=float, default=0.8)
    parser.add_argument("--w-fused", type=float, default=1.2)
    parser.add_argument("--w-grad", type=float, default=1.2)
    parser.add_argument("--w-region", type=float, default=0.15)
    parser.add_argument("--w-bg", type=float, default=0.35)
    parser.add_argument("--w-smooth", type=float, default=0.03)
    parser.add_argument("--w-residual", type=float, default=0.02)
    return parser.parse_args()


def load_compatible_state_dict(model, state_dict):
    model_state = model.state_dict()
    loaded = {}
    skipped = []
    for key, value in state_dict.items():
        if key in model_state and tuple(model_state[key].shape) == tuple(value.shape):
            loaded[key] = value
        else:
            skipped.append(key)
    model_state.update(loaded)
    model.load_state_dict(model_state)
    return loaded, skipped


def make_loader(args, train):
    root = Path(args.root)
    split = args.train_split if train else args.val_split
    dataset = RegFuseDataset(
        ir_dir=root / args.ir_dir,
        vi_dir=root / args.vi_dir,
        split_file=root / split,
        pseudo_root=root / args.pseudo_root,
        crop_size=args.crop_size,
        train=train,
        on_the_fly_pseudo=args.on_the_fly_pseudo,
        block_size=args.block_size,
        smooth_radius=args.smooth_radius,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=train,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=train,
    )


def average_logs(logs):
    out = {}
    for log in logs:
        for key, value in log.items():
            out.setdefault(key, 0.0)
            out[key] += float(value)
    for key in out:
        out[key] /= max(len(logs), 1)
    return out


def run_epoch(model, loader, optimizer, scaler, args, train):
    model.train(train)
    logs = []
    for batch in loader:
        x = batch["x"].to(args.device, non_blocking=True)
        ir = batch["ir"].to(args.device, non_blocking=True)
        vi = batch["vi"].to(args.device, non_blocking=True)
        mask = batch["mask"].to(args.device, non_blocking=True)
        alpha = batch["alpha"].to(args.device, non_blocking=True)
        fused_teacher = batch["fused_teacher"].to(args.device, non_blocking=True)

        with torch.set_grad_enabled(train):
            with autocast(enabled=args.amp):
                seg_logits, alpha_pred, detail_gate, residual_pred = model(x)
                loss, log = total_loss(
                    seg_logits,
                    alpha_pred,
                    detail_gate,
                    residual_pred,
                    mask,
                    alpha,
                    fused_teacher,
                    x,
                    ir,
                    vi,
                    w_seg=args.w_seg,
                    w_alpha=args.w_alpha,
                    w_fused=args.w_fused,
                    w_grad=args.w_grad,
                    w_region=args.w_region,
                    w_bg=args.w_bg,
                    w_smooth=args.w_smooth,
                    w_residual=args.w_residual,
                )
            if train:
                if not torch.isfinite(loss):
                    print("Skip non-finite loss batch.")
                    logs.append(log)
                    continue
                optimizer.zero_grad(set_to_none=True)
                if args.amp:
                    scaler.scale(loss).backward()
                    if args.grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()
        logs.append(log)
    return average_logs(logs)


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_val, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "best_val": best_val,
            "args": vars(args),
        },
        path,
    )


def main():
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    train_loader = make_loader(args, train=True)
    val_loader = make_loader(args, train=False)
    model = RegFuseUNet(in_channels=8, num_classes=5, base_channels=args.base_channels).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup = max(int(args.warmup_epochs), 0)
    if warmup > 0:
        scheduler = SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup),
                CosineAnnealingLR(optimizer, T_max=max(args.epochs - warmup, 1), eta_min=args.min_lr),
            ],
            milestones=[warmup],
        )
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1), eta_min=args.min_lr)
    scaler = GradScaler(enabled=args.amp)
    start_epoch = 1
    best_val = float("inf")

    if args.resume:
        ckpt = torch.load(Path(args.resume), map_location=args.device)
        loaded, skipped = load_compatible_state_dict(model, ckpt["model"])
        print(f"Resumed model from: {args.resume}")
        print(f"Loaded compatible tensors: {len(loaded)}, skipped tensors: {len(skipped)}")
        if "optimizer" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
            except ValueError:
                print("Optimizer state was not loaded because model shape changed.")
        if ckpt.get("scheduler") is not None:
            try:
                scheduler.load_state_dict(ckpt["scheduler"])
            except (ValueError, KeyError):
                print("Scheduler state was not loaded.")
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        if args.keep_best:
            best_val = float(ckpt.get("best_val", best_val))
        else:
            print("Reset best_val for the current training objective.")

    print(f"Device: {args.device}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Model parameters: {count_parameters(model) / 1e6:.2f} M")
    print(f"Batch size: {args.batch_size}, crop size: {args.crop_size}, base channels: {args.base_channels}")

    for epoch in range(start_epoch, args.epochs + 1):
        train_log = run_epoch(model, train_loader, optimizer, scaler, args, train=True)
        val_log = run_epoch(model, val_loader, optimizer, scaler, args, train=False)
        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"lr={optimizer.param_groups[0]['lr']:.2e} "
            f"train_loss={train_log['loss']:.4f} val_loss={val_log['loss']:.4f} "
            f"val_seg={val_log['seg']:.4f} val_alpha={val_log['alpha']:.4f} "
            f"val_fused={val_log['fused']:.4f} val_grad={val_log['grad']:.4f} "
            f"val_region={val_log['region']:.4f} val_bg={val_log['bg']:.4f} "
            f"val_smooth={val_log['smooth']:.4f} val_residual={val_log['residual']:.4f}"
        )
        scheduler.step()
        save_checkpoint(save_dir / "last.pth", model, optimizer, scheduler, epoch, best_val, args)
        if val_log["loss"] < best_val:
            best_val = val_log["loss"]
            save_checkpoint(save_dir / "best.pth", model, optimizer, scheduler, epoch, best_val, args)
            print(f"Saved best checkpoint: {save_dir / 'best.pth'}")


if __name__ == "__main__":
    main()
