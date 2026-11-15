"""Streamlit UI. Runs the pipeline in-process, or calls the FastAPI service when RYFC_API_URL is set."""

import base64
import json
import os
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ryfc import pipeline, report  # noqa: E402
from ryfc.config import PACKAGES, ROOM_CLASSES, SAMPLES, load_yaml  # noqa: E402
from ryfc.report import CAT_NAMES  # noqa: E402

API = os.getenv("RYFC_API_URL")
MARKET = load_yaml("market.yaml")

st.set_page_config(page_title="Finishing Cost & Rental Yield", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=GFS+Didot&display=swap');
:root { --ink:#1d1d1b; --muted:#4a4a46; --line:#d9d6cf; --paper:#faf9f6; --accent:#7a3e1d; }
html, body, .stApp { background: var(--paper); color: var(--ink); }
h1, h2, h3, .display { font-family: 'GFS Didot', 'Didot', Georgia, serif !important; font-weight: 400 !important;
  color: var(--ink); letter-spacing: 0; }
.stApp, p, li, label, .stMarkdown, .stDataFrame, input, select, textarea, button {
  font-family: 'Times New Roman', Times, serif; }
label, .stMetric label, [data-testid="stMetricLabel"] p { font-weight: 700 !important; color: var(--muted) !important; }
[data-testid="stMetricValue"] { font-family: 'Times New Roman', Times, serif; font-weight: 700; color: var(--ink); }
.card { border: 1px solid var(--line); background: #fff; padding: 14px 16px; border-radius: 4px; }
.card h4 { font-family: 'GFS Didot', Georgia, serif; font-weight: 400; margin: 0 0 6px 0; font-size: 1.25rem; }
.card .big { font-family: 'Times New Roman', serif; font-weight: 700; font-size: 1.35rem; }
.card .row { display:flex; justify-content:space-between; border-top:1px solid var(--line); padding:4px 0;
  font-family: 'Times New Roman', serif; color: var(--ink); }
.card .row b { font-weight: 700; }
.rec { border: 2px solid var(--accent); }
.tag { color: var(--accent); font-weight: 700; font-family: 'Times New Roman', serif; }
.note { color: var(--muted); font-size: 0.95rem; }
</style>
""", unsafe_allow_html=True)


def egp(v):
    return f"EGP {v:,.0f}"


def call_analyze(data, name, unit, use_ocr):
    if not API:
        return pipeline.analyze(data, name, pipeline.UnitInfo(**unit), use_ocr)
    import httpx
    r = httpx.post(f"{API}/analyze", files={"file": (name, data)},
                   data={"unit": json.dumps(unit), "use_ocr": str(use_ocr).lower()}, timeout=120)
    if r.status_code != 200:
        raise ValueError(r.json().get("detail", r.text))
    return r.json()


def call_evaluate(rooms, unit):
    if not API:
        return pipeline.evaluate(rooms, pipeline.UnitInfo(**unit))
    import httpx
    r = httpx.post(f"{API}/evaluate", json={"rooms": rooms, "unit": unit}, timeout=60)
    r.raise_for_status()
    return r.json()


st.markdown("<h1>Finishing Cost &amp; Rental Yield</h1>", unsafe_allow_html=True)
st.markdown("<p class='note'>Upload the floor plan of an unfinished unit. The estimator reads the rooms, prices "
            "three finishing packages, predicts the rent range for each, and shows which one pays back best.</p>",
            unsafe_allow_html=True)

with st.sidebar:
    st.markdown("<h3>Unit</h3>", unsafe_allow_html=True)
    area = st.selectbox("Area", list(MARKET["areas"]))
    unit_type = st.selectbox("Unit type", list(MARKET["unit_type"]))
    tier = st.selectbox("Compound tier", ["A", "B", "C"], index=1, help="A = premium compound, C = economy")
    area_mode = st.radio("Area entered as", ["BUA (sales area)", "Net internal area"], horizontal=True)
    size = st.number_input("Area (m2)", 40.0, 1000.0, 165.0, 5.0)
    floor = st.number_input("Floor", 0, 40, 3)
    view = st.selectbox("View", list(MARKET["view"]), index=2)
    furnished = st.checkbox("Will be rented furnished")
    amenities = st.slider("Compound amenities (club, pool, gym, security, retail)", 0, 5, 3)
    delivery = st.selectbox("Delivered as", ["semi_finished", "core_shell"],
                            format_func=lambda s: s.replace("_", " ").title())
    contractor = st.selectbox("Contractor tier", ["budget", "mid", "top"], index=1)
    price = st.number_input("Purchase price (EGP, 0 = estimate)", 0, 200_000_000, 0, 100_000)
    with st.expander("Yield assumptions"):
        d = MARKET["yield_defaults"]
        vac = st.number_input("Vacancy (months/year)", 0.0, 6.0, float(d["vacancy_months"]), 0.5)
        fee = st.number_input("Management fee (% of rent)", 0.0, 30.0, d["management_fee"] * 100, 1.0) / 100
        maint = st.number_input("Maintenance (% of rent)", 0.0, 20.0, d["maintenance_rate"] * 100, 1.0) / 100
        disc = st.number_input("Discount rate (%)", 5.0, 40.0, d["discount_rate"] * 100, 1.0) / 100
        horizon = st.number_input("Horizon (years)", 3, 25, int(d["horizon_years"]))
    use_ocr = st.checkbox("Read room names and dimensions (OCR)", True, key="ocr")

unit = {"area": area, "unit_type": unit_type, "compound_tier": tier, "floor": int(floor), "view": view,
        "furnished": furnished, "amenities": amenities, "delivery": delivery, "contractor_tier": contractor,
        "purchase_price": price or None, "vacancy_months": vac, "management_fee": fee, "maintenance_rate": maint,
        "discount_rate": disc, "horizon_years": int(horizon),
        ("bua_m2" if area_mode.startswith("BUA") else "net_area_m2"): size}

c1, c2 = st.columns([2, 1])
with c1:
    up = st.file_uploader("Floor plan (PNG, JPG or PDF)", type=["png", "jpg", "jpeg", "pdf"])
with c2:
    samples = sorted(SAMPLES.glob("plan_*.*")) if SAMPLES.exists() else []
    pick = st.selectbox("...or try a sample plan", ["-"] + [p.name for p in samples], key="sample")

demo = st.query_params.get("demo")  # shareable link: ?demo=plan_2.png runs a sample straight away
if demo and pick == "-" and (SAMPLES / demo).is_file() and "result" not in st.session_state:
    pick = demo
source = (up.getvalue(), up.name) if up else ((SAMPLES / pick).read_bytes(), pick) if pick != "-" else None
if source and (st.button("Analyze plan", type="primary", key="analyze") or (demo and "result" not in st.session_state)):
    with st.spinner("Reading plan, pricing packages, predicting rent..."):
        try:
            st.session_state.result = call_analyze(source[0], source[1], unit, use_ocr)
            st.session_state.unit = unit
        except ValueError as e:
            st.error(str(e))

res = st.session_state.get("result")
if not res:
    st.stop()

left, right = st.columns([1.1, 1])
with left:
    st.markdown("<h3>Detected rooms</h3>", unsafe_allow_html=True)
    st.image(base64.b64decode(res["annotated_png"]), width="stretch")
    st.markdown(f"<p class='note'>Scale from {res['scale']['method']} · net area "
                f"{res['net_area_m2']:.1f} m2 · {res['latency_s']:.1f} s</p>", unsafe_allow_html=True)
    for w in res.get("warnings", []):
        st.warning(w)
with right:
    st.markdown("<h3>Check and correct</h3>", unsafe_allow_html=True)
    df = pd.DataFrame([{"#": r["id"], "Room": r["type"], "Area m2": round(r["area_m2"], 1),
                        "Confidence": round(r["confidence"], 2), "Check": "yes" if r["flagged"] else "",
                        "perimeter_m": r["perimeter_m"]} for r in res["rooms"]])
    edited = st.data_editor(df, hide_index=True, num_rows="dynamic", width="stretch",
                            column_config={"Room": st.column_config.SelectboxColumn(options=ROOM_CLASSES),
                                           "Confidence": st.column_config.NumberColumn(disabled=True),
                                           "Check": st.column_config.TextColumn(disabled=True),
                                           "#": st.column_config.NumberColumn(disabled=True),
                                           "perimeter_m": None})
    rooms = [{"type": r["Room"], "area_m2": float(r["Area m2"]),
              "perimeter_m": None if pd.isna(r.get("perimeter_m")) else float(r["perimeter_m"])}
             for r in edited.to_dict("records") if r.get("Room") and r.get("Area m2")]
    st.markdown("<p class='note'>Edits update the costs and yield below.</p>", unsafe_allow_html=True)

fin = call_evaluate(rooms, st.session_state.unit)
res["financials"] = fin
best = fin["recommended_package"]

st.markdown("<h2>Finishing packages</h2>", unsafe_allow_html=True)
cols = st.columns(3)
for col, p in zip(cols, PACKAGES):
    x = fin["packages"][p]
    a, y, r = x["cost"]["adjusted"], x["yield"], x["rent"]
    pb = f"{y['finishing_payback_years']:.1f} years" if y["finishing_payback_years"] else "over 50 years"
    with col:
        st.markdown(f"""<div class="card {'rec' if p == best else ''}">
<h4>{p.title()}</h4>{'<span class="tag">Best 10-year value</span>' if p == best else '&nbsp;'}
<div class="big">{egp(a['estimate'])}</div>
<div class="note">{a['estimate'] / fin['net_area_m2']:,.0f} EGP/m2 · range {a['low'] / 1e6:.2f}-{a['high'] / 1e6:.2f}M</div>
<div class="row"><span>Rent / month</span><b>{r['low']:,.0f} - {r['high']:,.0f}</b></div>
<div class="row"><span>Median rent</span><b>{r['median']:,.0f}</b></div>
<div class="row"><span>Gross / net yield</span><b>{y['gross_yield'] * 100:.1f}% / {y['net_yield'] * 100:.1f}%</b></div>
<div class="row"><span>Finishing payback</span><b>{pb}</b></div>
<div class="row"><span>NPV of finishing</span><b>{egp(y['npv_of_finishing'])}</b></div>
</div>""", unsafe_allow_html=True)

st.markdown("<p class='note'>Purchase price " + egp(fin["purchase_price"]) +
            (" (estimated from area prices; enter yours in the sidebar)" if fin["purchase_price_estimated"] else "") +
            ". Yields use price + finishing. NPV = present value of net rent over the horizon minus finishing cost.</p>",
            unsafe_allow_html=True)

b1, b2 = st.columns([1.3, 1])
FONT = dict(family="Times New Roman, Times, serif", color="#1d1d1b", size=14)
with b1:
    st.markdown("<h3>Cost breakdown</h3>", unsafe_allow_html=True)
    fig = go.Figure()
    shades = ["#3b3a36", "#6b675e", "#8f8a7d", "#b3ad9e", "#7a3e1d", "#a8643f", "#cf9a78", "#e2cdb9"]
    for cat, shade in zip(CAT_NAMES, shades):
        vals = [fin["packages"][p]["cost"]["breakdown"][cat] * fin["packages"][p]["cost"]["adjusted"]["estimate"]
                / fin["packages"][p]["cost"]["total"] for p in PACKAGES]
        fig.add_bar(name=CAT_NAMES[cat], x=[p.title() for p in PACKAGES], y=vals, marker_color=shade,
                    hovertemplate="%{x}: EGP %{y:,.0f}<extra>" + CAT_NAMES[cat] + "</extra>")
    fig.update_layout(barmode="stack", height=380, margin=dict(l=10, r=10, t=10, b=10), font=FONT,
                      plot_bgcolor="#faf9f6", paper_bgcolor="#faf9f6", legend=dict(font=FONT),
                      yaxis=dict(tickformat=",.0f", gridcolor="#e6e3dc", title="EGP"))
    st.plotly_chart(fig, width="stretch")
with b2:
    st.markdown(f"<h3>Line items · {best}</h3>", unsafe_allow_html=True)
    items = pd.DataFrame(fin["packages"][best]["cost"]["items"])
    items = items[["item", "qty", "unit", "total"]].rename(columns=str.title)
    st.dataframe(items.style.format({"Total": "{:,.0f}", "Qty": "{:,.1f}"}), hide_index=True,
                 width="stretch", height=380)

st.markdown("<h3>What drives the rent</h3>", unsafe_allow_html=True)
drv = fin["packages"]["standard"]["rent"]["drivers"]
st.markdown(" · ".join(f"<b>{d['feature'].replace('_', ' ')}</b> {d['effect_pct']:+.1f}%" for d in drv),
            unsafe_allow_html=True)

pdf = report.build(res, st.session_state.unit)
st.download_button("Download PDF report", pdf, "finishing-yield-report.pdf", "application/pdf", type="primary")
