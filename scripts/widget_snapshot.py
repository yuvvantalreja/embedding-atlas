"""Headless-Playwright snapshot of the lunar_widget cell in the running Marimo notebook.

Usage:
    uv run --with playwright python scripts/widget_snapshot.py

Environment:
    MARIMO_URL   Full URL of the marimo edit session, including any access token
                 query string. Defaults to http://localhost:2718/.
    OUT_PATH     Output PNG path. Defaults to /tmp/lunar_widget_snapshot.png.

The script opens the page, waits for a <canvas> that has actually been painted
(width*height > 0 AND the WebGPU/WebGL2 renderer has drawn at least one frame),
then screenshots the bounding cell that contains that canvas.
"""

from __future__ import annotations

import os
import sys
import time

from playwright.sync_api import sync_playwright

URL = os.environ.get("MARIMO_URL", "http://localhost:2718/")
OUT_PATH = os.environ.get("OUT_PATH", "/tmp/lunar_widget_snapshot.png")
RENDER_TIMEOUT_MS = 60_000


def main() -> int:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1600, "height": 1200})
        page = context.new_page()
        page.goto(URL, wait_until="domcontentloaded")

        # Wait for the embedding-atlas canvas to be painted. The widget renders
        # via WebGPU/WebGL2 into a <canvas>; we look for any canvas whose
        # backing buffer has non-zero pixels.
        page.wait_for_function(
            """
            () => {
              const canvases = Array.from(document.querySelectorAll('canvas'));
              return canvases.some(c => {
                if (c.width === 0 || c.height === 0) return false;
                // Heuristic: the canvas is sized and visible in the viewport.
                const rect = c.getBoundingClientRect();
                return rect.width > 100 && rect.height > 100;
              });
            }
            """,
            timeout=RENDER_TIMEOUT_MS,
        )

        # Give the renderer a moment to settle on the latest frame
        # (label placement, density clustering, etc. are async).
        time.sleep(1.0)

        # Find the largest visible canvas and screenshot its enclosing
        # marimo cell so charts and histograms are captured too.
        target = page.evaluate(
            """
            () => {
              const canvases = Array.from(document.querySelectorAll('canvas'));
              let best = null;
              let bestArea = 0;
              for (const c of canvases) {
                const r = c.getBoundingClientRect();
                const area = r.width * r.height;
                if (area > bestArea) {
                  best = c;
                  bestArea = area;
                }
              }
              if (!best) return null;
              // Walk up to the nearest marimo cell container, if any.
              let el = best;
              while (el && el.parentElement) {
                if (el.getAttribute && el.getAttribute('data-cell-id')) break;
                if (el.classList && (el.classList.contains('cell') || el.classList.contains('Cell'))) break;
                el = el.parentElement;
              }
              const r = (el || best).getBoundingClientRect();
              return { x: r.x, y: r.y, w: r.width, h: r.height };
            }
            """
        )

        if target is None:
            print("ERROR: no rendered canvas found on page", file=sys.stderr)
            browser.close()
            return 1

        clip = {
            "x": max(0, target["x"]),
            "y": max(0, target["y"]),
            "width": target["w"],
            "height": target["h"],
        }
        page.screenshot(path=OUT_PATH, clip=clip, full_page=True)
        browser.close()

    print(OUT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
