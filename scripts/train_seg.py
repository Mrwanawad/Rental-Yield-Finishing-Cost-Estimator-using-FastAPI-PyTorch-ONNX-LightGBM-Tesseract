"""Train the room-segmentation U-Net on on-the-fly synthetic plans, log to MLflow, keep best by val mIoU.

python scripts/train_seg.py --steps 6000 --batch 8
"""

import argparse
import time

import numpy as np
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader, Dataset

from ryfc.config import CLASSES, WORK
from ryfc.synth import generate
from ryfc.tracking import run

VAL_SEED = 10_000_000


class SynthPlans(Dataset):
    def __init__(self, n, seed0):
        self.n, self.seed0 = n, seed0

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        s = generate(self.seed0 + i)
        x = (255 - s.image).astype(np.float32)[None] / 255.0
        return torch.from_numpy(x), torch.from_numpy(s.mask.astype(np.int64))


def build_model(pretrained=True):
    return smp.Unet("mobilenet_v2", encoder_weights="imagenet" if pretrained else None, in_channels=1,
                    classes=len(CLASSES))


def confusion(pred, target, k):
    idx = target.flatten() * k + pred.flatten()
    return torch.bincount(idx, minlength=k * k).reshape(k, k)


def iou_from_conf(conf):
    conf = conf.double()
    inter = conf.diag()
    union = conf.sum(0) + conf.sum(1) - inter
    return (inter / union.clamp(min=1)).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--val", type=int, default=200)
    ap.add_argument("--init", help="checkpoint to fine-tune from")
    ap.add_argument("--seed0", type=int, default=0, help="first synthetic seed (use new seeds when fine-tuning)")
    ap.add_argument("--ckpt", default="seg_best.pt")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(pretrained=not args.init)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location="cpu"))
    model = model.to(dev)
    train = DataLoader(SynthPlans(args.steps * args.batch, args.seed0), batch_size=args.batch, shuffle=False,
                       num_workers=args.workers, persistent_workers=True, pin_memory=True)
    val = DataLoader(SynthPlans(args.val, VAL_SEED), batch_size=8, num_workers=2)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    ce = torch.nn.CrossEntropyLoss()
    dice = smp.losses.DiceLoss("multiclass")
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    best, t0 = -1.0, time.time()
    ckpt = WORK / args.ckpt
    ckpt.parent.mkdir(parents=True, exist_ok=True)

    with run("segmentation", vars(args) | {"encoder": "mobilenet_v2", "arch": "Unet"}) as log:
        model.train()
        for step, (x, y) in enumerate(train, 1):
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast(dev, enabled=dev == "cuda"):
                out = model(x)
                loss = ce(out, y) + dice(out, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            if step % 100 == 0:
                print(f"step {step} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
                log({"loss": loss.item()}, step)
            if step % 500 == 0 or step == args.steps:
                model.eval()
                conf = torch.zeros(len(CLASSES), len(CLASSES), dtype=torch.long, device=dev)
                with torch.no_grad(), torch.autocast(dev, enabled=dev == "cuda"):
                    for vx, vy in val:
                        p = model(vx.to(dev)).argmax(1)
                        conf += confusion(p, vy.to(dev), len(CLASSES))
                ious = iou_from_conf(conf)
                miou = float(ious[1:].mean())  # rooms + wall, outside excluded
                print(f"  val mIoU {miou:.4f} " + " ".join(f"{c}={v:.3f}" for c, v in zip(CLASSES, ious)), flush=True)
                log({"val_miou": miou, **{f"iou_{c}": float(v) for c, v in zip(CLASSES, ious)}}, step)
                if miou > best:
                    best = miou
                    torch.save(model.state_dict(), ckpt)
                model.train()
        log({"best_val_miou": best}, args.steps)
    print(f"best val mIoU {best:.4f} -> {ckpt}")


if __name__ == "__main__":
    main()
