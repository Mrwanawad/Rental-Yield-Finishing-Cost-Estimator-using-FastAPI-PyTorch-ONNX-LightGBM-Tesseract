"""Export the best checkpoint to models/seg.onnx and check parity with PyTorch."""

import argparse
import time

import numpy as np
import onnxruntime as ort
import torch

from ryfc.config import MODELS, SEG_SIZE, WORK
from ryfc.synth import generate
from train_seg import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="seg_best.pt")
    args = ap.parse_args()
    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(WORK / args.ckpt, map_location="cpu"))
    model.eval()
    x = torch.zeros(1, 1, SEG_SIZE, SEG_SIZE)
    out = MODELS / "seg.onnx"
    MODELS.mkdir(exist_ok=True)
    torch.onnx.export(model, x, str(out), input_names=["image"], output_names=["logits"], opset_version=17,
                      dynamo=False)
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    s = generate(123)
    xi = ((255 - s.image).astype(np.float32) / 255)[None, None]
    with torch.no_grad():
        ref = model(torch.from_numpy(xi)).numpy()
    t = time.perf_counter()
    got = sess.run(None, {"image": xi})[0]
    ms = (time.perf_counter() - t) * 1000
    agree = (ref.argmax(1) == got.argmax(1)).mean()
    print(f"{out} {out.stat().st_size / 1e6:.1f} MB | max abs diff {np.abs(ref - got).max():.2e} | "
          f"argmax agreement {agree:.5f} | CPU latency {ms:.0f} ms")


if __name__ == "__main__":
    main()
