import torch
import torch.nn.functional as F


LOSS_PROFILE = "regfuse_v2"


def dice_loss(logits, target, num_classes=5, eps=1e-6):
    prob = torch.softmax(logits, dim=1)
    target_oh = F.one_hot(target.clamp(0, num_classes - 1), num_classes=num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    inter = torch.sum(prob * target_oh, dims)
    denom = torch.sum(prob + target_oh, dims)
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def gradient_xy(x):
    dx = x[..., :, 1:] - x[..., :, :-1]
    dy = x[..., 1:, :] - x[..., :-1, :]
    return dx, dy


def gradient_loss(fused, ir, vi):
    fdx, fdy = gradient_xy(fused)
    idx, idy = gradient_xy(ir)
    vdx, vdy = gradient_xy(vi)
    target_dx = torch.where(torch.abs(idx) > torch.abs(vdx), idx, vdx)
    target_dy = torch.where(torch.abs(idy) > torch.abs(vdy), idy, vdy)
    return F.l1_loss(fdx, target_dx) + F.l1_loss(fdy, target_dy)


def alpha_smooth_loss(alpha):
    dx, dy = gradient_xy(alpha)
    return dx.abs().mean() + dy.abs().mean()


def box_blur_torch(x, kernel_size=5):
    pad = kernel_size // 2
    return F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=pad)


def compose_fusion(x, ir, vi, alpha_pred, detail_gate, residual_pred, residual_scale=0.10, detail_scale=0.55):
    hot = x[:, 5:6]
    alpha = alpha_pred.clamp(0.0, 1.0)
    base = (1.0 - alpha) * vi + alpha * ir

    detail_i = ir - box_blur_torch(ir, kernel_size=5)
    detail_v = vi - box_blur_torch(vi, kernel_size=5)
    thermal_gate = (0.65 * alpha + 0.35 * hot).clamp(0.0, 1.0)
    target_detail = 0.55 * detail_i + 0.45 * detail_v
    detail = thermal_gate * target_detail + (1.0 - thermal_gate) * detail_v

    fused = base + detail_scale * detail_gate * detail + residual_scale * residual_pred
    return fused.clamp(0.0, 1.0)


def visible_background_loss(fused, vi, alpha_gt):
    background = (alpha_gt < 0.12).float()
    return masked_l1(fused, vi, background)


def masked_l1(a, b, mask):
    mask = mask.float()
    denom = mask.sum().clamp_min(1.0)
    return torch.sum(torch.abs(a - b) * mask) / denom


def region_loss(fused, ir, vi, mask):
    mask = mask.unsqueeze(1)
    loss = 0.0
    loss = loss + masked_l1(fused, ir, mask == 1)
    loss = loss + masked_l1(fused, 0.5 * (ir + vi), mask == 0)

    fdx, fdy = gradient_xy(fused)
    idx, idy = gradient_xy(ir)
    vdx, vdy = gradient_xy(vi)
    mask2 = mask[..., :, 1:]
    mask2y = mask[..., 1:, :]
    loss = loss + masked_l1(fdx, vdx, mask2 == 2)
    loss = loss + masked_l1(fdy, vdy, mask2y == 2)

    max_dx = torch.where(torch.abs(idx) > torch.abs(vdx), idx, vdx)
    max_dy = torch.where(torch.abs(idy) > torch.abs(vdy), idy, vdy)
    loss = loss + masked_l1(fdx, max_dx, mask2 == 4)
    loss = loss + masked_l1(fdy, max_dy, mask2y == 4)
    return loss


def total_loss(
    seg_logits,
    alpha_pred,
    detail_gate,
    residual_pred,
    mask_gt,
    alpha_gt,
    fused_teacher,
    x,
    ir,
    vi,
    w_seg=1.0,
    w_alpha=0.8,
    w_fused=1.2,
    w_grad=1.2,
    w_region=0.15,
    w_bg=0.35,
    w_smooth=0.03,
    w_residual=0.02,
):
    ce = F.cross_entropy(seg_logits, mask_gt)
    dice = dice_loss(seg_logits, mask_gt, num_classes=seg_logits.shape[1])
    seg = ce + dice
    alpha = F.l1_loss(alpha_pred, alpha_gt)
    fused = compose_fusion(x, ir, vi, alpha_pred, detail_gate, residual_pred)
    fused_loss = F.l1_loss(fused, fused_teacher)
    grad = gradient_loss(fused, ir, vi)
    region = region_loss(fused, ir, vi, mask_gt)
    bg = visible_background_loss(fused, vi, alpha_gt)
    smooth = alpha_smooth_loss(alpha_pred)
    residual = residual_pred.abs().mean()
    total = (
        w_seg * seg
        + w_alpha * alpha
        + w_fused * fused_loss
        + w_grad * grad
        + w_region * region
        + w_bg * bg
        + w_smooth * smooth
        + w_residual * residual
    )
    return total, {
        "loss": float(total.detach().cpu()),
        "seg": float(seg.detach().cpu()),
        "alpha": float(alpha.detach().cpu()),
        "fused": float(fused_loss.detach().cpu()),
        "grad": float(grad.detach().cpu()),
        "region": float(region.detach().cpu()),
        "bg": float(bg.detach().cpu()),
        "smooth": float(smooth.detach().cpu()),
        "residual": float(residual.detach().cpu()),
    }
