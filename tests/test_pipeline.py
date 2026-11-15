import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from ryfc import cost, ocr, pipeline, report, synth, vision
from ryfc.config import CLASS_ID, MODELS, PACKAGES, SAMPLES
from ryfc.yieldcalc import YieldInputs, compute

ROOMS = [
    {"type": "living", "area_m2": 45.0}, {"type": "bedroom", "area_m2": 16.0}, {"type": "bedroom", "area_m2": 14.0},
    {"type": "bedroom", "area_m2": 13.0}, {"type": "kitchen", "area_m2": 12.0}, {"type": "bathroom", "area_m2": 6.0},
    {"type": "bathroom", "area_m2": 3.0}, {"type": "corridor", "area_m2": 8.0}, {"type": "balcony", "area_m2": 7.0},
]
has_models = (MODELS / "seg.onnx").exists() and (MODELS / "rent_q50.txt").exists()


def test_synth_ground_truth_is_consistent():
    s = synth.generate(3)
    assert s.image.shape == s.mask.shape == (512, 512)
    assert {CLASS_ID["wall"], CLASS_ID["living"], CLASS_ID["bedroom"]} <= set(np.unique(s.mask))
    assert 60 < s.net_area_m2 < 300
    assert abs(sum(r["area_m2"] for r in s.rooms) - s.net_area_m2) < 1e-6


def test_synth_is_deterministic():
    assert (synth.generate(11).image == synth.generate(11).image).all()


def test_cost_rules_order_and_band():
    totals = {p: cost.rules(ROOMS, p)["per_m2"] for p in PACKAGES}
    assert totals["basic"] < totals["standard"] < totals["premium"]
    assert 5_000 < totals["basic"] < 9_000
    assert 8_500 < totals["standard"] < 14_000
    assert 16_000 < totals["premium"] < 30_000


def test_semi_finished_is_cheaper_than_core_shell():
    assert cost.rules(ROOMS, "standard", "semi_finished")["total"] < cost.rules(ROOMS, "standard", "core_shell")["total"]


def test_breakdown_sums_to_total():
    c = cost.rules(ROOMS, "standard")
    assert sum(c["breakdown"].values()) == pytest.approx(c["total"])
    assert sum(i["total"] for i in c["items"]) == pytest.approx(c["total"])


def test_ac_sizing():
    assert cost.ac_hp(12) == [1.5]
    assert cost.ac_hp(20) == [2.25]
    assert len(cost.ac_hp(60)) == 2


def test_yield_math():
    y = compute(YieldInputs(purchase_price=1_000_000, finishing_cost=200_000, monthly_rent=10_000,
                            vacancy_months=0, management_fee=0, maintenance_rate=0, annual_rent_growth=0))
    assert y["gross_yield"] == pytest.approx(0.1)
    assert y["net_yield"] == pytest.approx(0.1)
    assert y["finishing_payback_years"] == pytest.approx(200_000 / 120_000)
    assert y["total_payback_years"] == pytest.approx(10.0)


def test_payback_never_reached():
    y = compute(YieldInputs(purchase_price=1e9, finishing_cost=1e9, monthly_rent=1, annual_rent_growth=0))
    assert y["total_payback_years"] is None


def test_ocr_word_classification():
    assert ocr.classify_word("BEDROOM") == "bedroom"
    assert ocr.classify_word("EEDROOM") == "bedroom"
    assert ocr.classify_word("W.C") == "bathroom"
    assert ocr.classify_word("Reception") == "living"
    assert ocr.classify_word("Room") is None
    assert ocr.DIMS.search("3.68 x 4.84").groups() == ("3.68", "4.84")


def test_rooms_from_perfect_probs():
    s = synth.generate(5)
    probs = np.eye(8, dtype=np.float32)[s.mask].transpose(2, 0, 1)
    rooms, inst = vision.rooms_from_probs(probs)
    vision.apply_scale(rooms, s.ppm)
    assert sum(r["area_m2"] for r in rooms) == pytest.approx(s.net_area_m2, rel=0.03)
    assert {r["type"] for r in rooms} >= {"living", "bedroom", "kitchen", "bathroom"}
    assert inst.max() == len(rooms)


def test_scale_from_dims():
    rooms = [{"pixels": 400 * 12, "fill_ratio": 0.95, "ocr_dims": [3.0, 4.0], "bbox": [0, 0, 60, 80]},
             {"pixels": 400 * 6, "fill_ratio": 0.95, "ocr_dims": [2.0, 3.0], "bbox": [0, 0, 40, 60]},
             # stray label from another room: wrong shape, rejected
             {"pixels": 400 * 6, "fill_ratio": 0.95, "ocr_dims": [9.0, 1.5], "bbox": [0, 0, 40, 60]}]
    assert vision.scale_from_dims(rooms) == pytest.approx(20.0)


@pytest.mark.skipif(not has_models, reason="trained models not present")
def test_end_to_end_sample_plan():
    gt = json.loads((SAMPLES / "ground_truth.json").read_text())["plan_2.png"]
    unit = pipeline.UnitInfo(net_area_m2=gt["net_area_m2"])
    r = pipeline.analyze((SAMPLES / "plan_2.png").read_bytes(), "plan_2.png", unit, use_ocr=False)
    assert r["net_area_m2"] == pytest.approx(gt["net_area_m2"], rel=1e-6)
    beds = sum(x["type"] == "bedroom" for x in r["rooms"])
    assert beds == sum(x["type"] == "bedroom" for x in gt["rooms"])
    f = r["financials"]
    assert f["recommended_package"] in PACKAGES
    for p in PACKAGES:
        rent = f["packages"][p]["rent"]
        assert rent["low"] < rent["median"] < rent["high"]
    assert report.build(r, unit.model_dump())[:4] == b"%PDF"


@pytest.mark.skipif(not has_models, reason="trained models not present")
def test_api_contract():
    from ryfc.api import app

    c = TestClient(app)
    assert c.get("/health").json()["status"] == "ok"
    data = (SAMPLES / "plan_1.pdf").read_bytes()
    r = c.post("/analyze", files={"file": ("plan_1.pdf", data, "application/pdf")},
               data={"unit": json.dumps({"bua_m2": 220, "area": "Sheikh Zayed"}), "use_ocr": "false"})
    assert r.status_code == 200, r.text
    body = r.json()
    rooms = [{"type": x["type"], "area_m2": x["area_m2"]} for x in body["rooms"]]
    e = c.post("/evaluate", json={"rooms": rooms, "unit": {"bua_m2": 220, "area": "Sheikh Zayed"}})
    assert e.status_code == 200
    assert c.post("/analyze", files={"file": ("x.png", b"not an image")}, data={"unit": "{}"}).status_code == 422
    assert c.post("/report", json={"result": body, "unit": {"bua_m2": 220}}).content[:4] == b"%PDF"
