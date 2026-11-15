from pathlib import Path

import pytest

from ryfc.config import MODELS

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")
pytest.importorskip("streamlit")


@pytest.mark.skipif(not (MODELS / "seg.onnx").exists(), reason="trained models not present")
def test_ui_sample_flow():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception
    at.selectbox(key="sample").select("plan_2.png").run()
    at.checkbox(key="ocr").uncheck().run()  # OCR off keeps the test fast
    at.button(key="analyze").click().run()
    assert not at.exception, at.exception
    html = " ".join(m.value for m in at.markdown)
    assert "Finishing packages" in html
    assert "Best 10-year value" in html
    assert len(at.get("data_editor")) == 1 or at.dataframe
