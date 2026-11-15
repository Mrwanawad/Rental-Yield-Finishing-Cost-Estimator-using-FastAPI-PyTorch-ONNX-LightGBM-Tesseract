"""README demo: floor plan in, rooms + finishing packages + yield + PDF out.

Run the app first:  uvicorn index:app --port 8020
Then:  python docs/demo/scene.py
       python ~/.claude/skills/demo-gif/scripts/compose.py docs/demo/build --out docs/demo.gif
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.expanduser("~/.claude/skills/demo-gif/scripts"))
from director import Director, run  # noqa: E402

URL = os.getenv("DEMO_URL", "http://localhost:8020/")


async def scene(d: Director):
    p = d.page
    await d.goto(URL, settle=1.0)
    await p.locator(".samples img").first.wait_for()

    await d.caption("Enter the unit's sales area", step=1)
    await d.type(p.locator("#size"), "274")

    await d.caption("Upload the floor plan", step=2)
    await d.click(p.locator(".samples img").nth(1))
    await d.fast(p.locator("#out").wait_for(state="visible", timeout=120_000), label="Reading the plan")
    await d.hold(0.4)

    await d.caption("Rooms and areas, ready to correct", step=3)
    await d.focus(p.locator("#out .two").first, zoom=1.3)
    await d.hold(1.6)
    await d.unfocus()

    await d.caption("Three packages: cost, rent, yield", step=4)
    await d.scroll(520)
    await d.focus(p.locator("#cards"), zoom=1.35)
    await d.spotlight(p.locator(".card.rec"))
    b = await p.locator(".card.rec").bounding_box()
    await d.move((b["x"] + b["width"] - 24, b["y"] + b["height"] - 18))
    await d.hold(2.0)
    await d.clear_spotlight()
    await d.unfocus()

    await d.caption("Download the PDF report", step=5)
    await d.click(p.get_by_role("button", name="Download PDF report"))
    await d.hold(1.6)


run(scene, out=str(Path(__file__).with_name("build")), title="Finishing Cost & Rental Yield",
    subtitle="From an unfinished unit's floor plan to the package that pays back best",
    url_label="rental-yield-finishing-cost.vercel.app")
