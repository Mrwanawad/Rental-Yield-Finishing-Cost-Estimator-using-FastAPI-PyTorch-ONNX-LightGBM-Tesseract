"""Downloadable PDF report (fpdf2, plain black and white)."""

import base64
import io
from datetime import date

from fpdf import FPDF

from ryfc.config import PACKAGES

CAT_NAMES = {"flooring": "Flooring", "plaster_paint_ceiling": "Plaster, paint & ceiling", "kitchen": "Kitchen",
             "bathrooms": "Bathrooms & plumbing", "electrical": "Electrical", "ac": "Air conditioning",
             "doors": "Doors", "contingency": "Contingency & margin"}


def egp(v):
    return f"EGP {v:,.0f}"


def pct(v):
    return f"{v * 100:.1f}%"


def years(v):
    return f"{v:.1f} y" if v is not None else "> 50 y"


class _PDF(FPDF):
    def header(self):
        self.set_font("Times", "B", 9)
        self.cell(0, 6, "Rental Yield & Finishing Cost Estimate", align="L")
        self.cell(0, 6, date.today().isoformat(), align="R", new_x="LMARGIN", new_y="NEXT")
        self.line(10, 17, 200, 17)
        self.ln(3)

    def footer(self):
        self.set_y(-12)
        self.set_font("Times", "", 8)
        self.cell(0, 6, f"Page {self.page_no()} - estimates, not a contractor quote or valuation", align="C")


def _table(pdf, header, rows, widths, bold_first=False):
    pdf.set_font("Times", "B", 9)
    for h, w in zip(header, widths):
        pdf.cell(w, 6, h, border="B", align="L")
    pdf.ln()
    pdf.set_font("Times", "", 9)
    for row in rows:
        for i, (v, w) in enumerate(zip(row, widths)):
            pdf.set_font("Times", "B" if bold_first and i == 0 else "", 9)
            pdf.cell(w, 5.5, str(v), align="L")
        pdf.ln()
    pdf.ln(3)


def build(result: dict, unit: dict) -> bytes:
    fin = result["financials"]
    best = fin["recommended_package"]
    pdf = _PDF()
    pdf.set_auto_page_break(True, 15)
    pdf.add_page()
    pdf.set_font("Times", "B", 16)
    pdf.cell(0, 9, f"{unit.get('unit_type', 'unit').title()} in {unit.get('area', '')}", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Times", "", 10)
    pdf.multi_cell(0, 5, f"Net area {fin['net_area_m2']:.1f} m2 (BUA {fin['bua_m2']:.0f} m2), {fin['bedrooms']} bedrooms, "
                         f"{fin['bathrooms']} bathrooms, floor {unit.get('floor', '-')}, compound tier "
                         f"{unit.get('compound_tier', '-')}, delivered {unit.get('delivery', '').replace('_', ' ')}. "
                         f"Purchase price {egp(fin['purchase_price'])}"
                         f"{' (estimated from area prices)' if fin['purchase_price_estimated'] else ''}.")
    pdf.ln(2)
    b = fin["packages"][best]
    pdf.set_font("Times", "B", 11)
    pdf.multi_cell(0, 5.5, f"Recommended: {best} finishing. Cost {egp(b['cost']['adjusted']['estimate'])}, "
                           f"rent {egp(b['rent']['low'])} - {egp(b['rent']['high'])} per month, net yield "
                           f"{pct(b['yield']['net_yield'])}, finishing paid back in "
                           f"{years(b['yield']['finishing_payback_years'])}.")
    pdf.ln(3)

    img = base64.b64decode(result["annotated_png"])
    pdf.image(io.BytesIO(img), x=45, w=120)
    pdf.ln(2)

    pdf.set_font("Times", "B", 12)
    pdf.cell(0, 7, "Detected rooms", new_x="LMARGIN", new_y="NEXT")
    rows = [[r["id"], r["type"].title(), f"{r['area_m2']:.1f}", f"{r['confidence']:.2f}",
             r.get("source", ""), "check" if r.get("flagged") else ""] for r in result["rooms"]]
    _table(pdf, ["#", "Room", "Area m2", "Confidence", "Source", "Flag"], rows, [10, 40, 25, 25, 25, 20])
    pdf.set_font("Times", "", 9)
    pdf.cell(0, 5, f"Scale from {result['scale']['method']}.", new_x="LMARGIN", new_y="NEXT")
    for w in result.get("warnings", []):
        pdf.multi_cell(0, 5, f"Note: {w}")

    pdf.add_page()
    pdf.set_font("Times", "B", 12)
    pdf.cell(0, 7, "Package comparison", new_x="LMARGIN", new_y="NEXT")
    rows = []
    for p in PACKAGES:
        x = fin["packages"][p]
        a = x["cost"]["adjusted"]
        rows.append([p.title(), egp(a["estimate"]), f"{a['estimate'] / fin['net_area_m2']:,.0f}",
                     f"{x['rent']['low']:,.0f} - {x['rent']['high']:,.0f}", pct(x["yield"]["gross_yield"]),
                     pct(x["yield"]["net_yield"]), years(x["yield"]["finishing_payback_years"]),
                     egp(x["yield"]["npv_of_finishing"])])
    _table(pdf, ["Package", "Finishing cost", "EGP/m2", "Rent/month (80%)", "Gross", "Net", "Payback", "NPV"],
           rows, [20, 30, 17, 35, 15, 15, 18, 35], bold_first=True)

    pdf.set_font("Times", "B", 12)
    pdf.cell(0, 7, f"Cost breakdown - {best}", new_x="LMARGIN", new_y="NEXT")
    adj = b["cost"]["adjusted"]
    scale = adj["estimate"] / b["cost"]["total"]
    rows = [[CAT_NAMES[k], egp(v * scale)] for k, v in b["cost"]["breakdown"].items() if v > 0]
    rows.append(["Total", egp(adj["estimate"])])
    rows.append(["Range (80%)", f"{egp(adj['low'])} - {egp(adj['high'])}"])
    _table(pdf, ["Category", "Cost"], rows, [70, 50])
    pdf.set_font("Times", "", 9)
    pdf.multi_cell(0, 5, f"Catalog rules x market adjustment {adj['multiplier']:.2f} x catalog inflation "
                         f"{adj['inflation_factor']:.3f}. Line items scaled proportionally.")
    pdf.ln(2)

    pdf.set_font("Times", "B", 12)
    pdf.cell(0, 7, "Rent drivers (standard finishing)", new_x="LMARGIN", new_y="NEXT")
    rows = [[d["feature"], f"{d['effect_pct']:+.1f}%"] for d in fin["packages"]["standard"]["rent"]["drivers"]]
    _table(pdf, ["Feature", "Effect vs. average"], rows, [50, 40])

    pdf.set_font("Times", "B", 12)
    pdf.cell(0, 7, "Method and assumptions", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Times", "", 9)
    y = b["yield"]
    pdf.multi_cell(0, 5, "Rooms: U-Net segmentation of the plan, room names from OCR where printed, areas scaled from "
                         "the entered unit area or printed dimensions. Cost: price catalog (EGP, installed) per room and "
                         "item, adjusted by a model of past jobs for region, floor, size and contractor tier. Rent: "
                         "quantile gradient boosting with conformal calibration (80% range). Yield: gross = annual rent / "
                         "(price + finishing); net after vacancy, management fee and maintenance; NPV of net rent over the "
                         f"horizon minus finishing cost. Net annual rent (median, {best}): {egp(y['net_annual_rent'])}.")
    return bytes(pdf.output())
