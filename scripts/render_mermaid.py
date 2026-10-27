import argparse
import re
from pathlib import Path

from playwright.sync_api import sync_playwright

REPO_ROOT = Path(__file__).resolve().parent.parent
PAGE = """<!doctype html><html><body style="margin:0;background:white"><div id="out" class="mermaid">{code}</div>
<script type="module">import mermaid from "{lib}"; mermaid.initialize({{startOnLoad: false, theme: "neutral"}}); await mermaid.run({{querySelector: "#out"}}); document.body.dataset.done = "1";</script></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="docs/presentation_diagrams.md")
    ap.add_argument("--out", default="report/presentation/mermaid")
    ap.add_argument("--chromium", default=None)
    ap.add_argument("--lib", default="https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.esm.min.mjs")
    args = ap.parse_args()
    text = (REPO_ROOT / args.src).read_text(encoding="utf-8")
    blocks = re.findall(r"^## (\d+)\. (.*?)\n\n```mermaid\n(.*?)```", text, re.S | re.M)
    out = REPO_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=args.chromium)
        page = browser.new_page(device_scale_factor=2, viewport={"width": 1400, "height": 900})
        for num, title, code in blocks:
            page.set_content(PAGE.format(code=code, lib=args.lib))
            page.wait_for_selector("body[data-done='1']", timeout=60000)
            target = out / f"mermaid_{int(num):02d}.png"
            page.locator("#out svg").screenshot(path=str(target))
            print(f"wrote {target} ({title})")
        browser.close()


if __name__ == "__main__":
    main()
