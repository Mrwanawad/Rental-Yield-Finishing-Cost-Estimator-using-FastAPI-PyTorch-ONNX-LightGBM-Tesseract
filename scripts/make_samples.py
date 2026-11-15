"""Write demo plans (PNG + one PDF) with their ground truth to samples/ and public/samples/."""

import json
import shutil

import cv2
from PIL import Image

from ryfc.config import ROOT, SAMPLES
from ryfc.synth import generate

SEEDS = [7, 21, 42, 105]


def main():
    SAMPLES.mkdir(exist_ok=True)
    truth = {}
    for i, seed in enumerate(SEEDS, 1):
        s = generate(seed, size=1024)
        name = f"plan_{i}.png"
        cv2.imwrite(str(SAMPLES / name), s.image)
        truth[name] = {"seed": seed, "net_area_m2": round(s.net_area_m2, 1),
                       "rooms": [{"type": r["type"], "area_m2": round(r["area_m2"], 1)} for r in s.rooms]}
    # same first plan as a PDF to exercise the PDF path
    Image.open(SAMPLES / "plan_1.png").convert("L").save(SAMPLES / "plan_1.pdf", resolution=144)
    truth["plan_1.pdf"] = truth["plan_1.png"]
    (SAMPLES / "ground_truth.json").write_text(json.dumps(truth, indent=2))
    public = ROOT / "public" / "samples"  # served statically by the web UI
    public.mkdir(parents=True, exist_ok=True)
    for f in SAMPLES.glob("plan_*.png"):
        shutil.copy(f, public / f.name)
    for k, v in truth.items():
        print(k, v["net_area_m2"], "m2", len(v["rooms"]), "rooms")


if __name__ == "__main__":
    main()
