#!/usr/bin/env python3
"""Generate assets/banner.png -- the repo's GitHub social-preview banner.

Run it directly (idempotent -- re-running regenerates byte-for-byte the same
image, since nothing here is randomized):

    python3 assets/make_banner.py

Pure Pillow, no network access, no external assets beyond the DejaVu fonts
already installed system-wide. Output is exactly 1280x640 (GitHub's
social-preview size).
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 1280, 640
OUT_PATH = Path(__file__).resolve().parent / "banner.png"

# ---------------------------------------------------------------------------
# Palette -- dark, GitHub-dark-friendly ground + two accents (teal, warm
# orange) plus off-white/muted text. No pure #fff large fills.
# ---------------------------------------------------------------------------
BG = "#0d1117"
GRID = "#161b22"
BOX_FILL = "#141a22"
CHIP_FILL = "#161c26"
TEAL = "#4fd1c5"          # primary accent -- SDK targets, "ADK" in wordmark
WARM = "#f0a35c"          # secondary accent -- source node, footer rule
TEXT_PRIMARY = "#e6edf3"  # off-white, not pure white
TEXT_MUTED = "#8b949e"
LINE_COLOR = "#3a6b6b"    # muted teal for fan-out connector lines

FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")
FONT_BOLD = FONT_DIR / "DejaVuSans-Bold.ttf"
FONT_REGULAR = FONT_DIR / "DejaVuSans.ttf"


def font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size)


def text_size(draw: ImageDraw.ImageDraw, text: str, f: ImageFont.FreeTypeFont) -> tuple[int, int]:
    l, t, r, b = draw.textbbox((0, 0), text, font=f)
    return r - l, b - t


def draw_centered(
    draw: ImageDraw.ImageDraw,
    cx: int,
    cy: int,
    text: str,
    f: ImageFont.FreeTypeFont,
    fill: str,
) -> tuple[int, int, int, int]:
    """Draw `text` centered at (cx, cy); return the bbox actually painted."""
    l, t, r, b = draw.textbbox((0, 0), text, font=f)
    w, h = r - l, b - t
    x = cx - w / 2 - l
    y = cy - h / 2 - t
    draw.text((x, y), text, font=f, fill=fill)
    return (x + l, y + t, x + r, y + b)


def rounded_rect(draw, box, radius, fill=None, outline=None, width=1):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def main() -> None:
    img = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(img)

    # -- subtle background grid -------------------------------------------------
    step = 40
    for x in range(0, WIDTH + 1, step):
        draw.line([(x, 0), (x, HEIGHT)], fill=GRID, width=1)
    for y in range(0, HEIGHT + 1, step):
        draw.line([(0, y), (WIDTH, y)], fill=GRID, width=1)

    # a soft radial-ish glow behind the headline, done as translucent ellipses
    glow = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    gdraw = ImageDraw.Draw(glow)
    for i, r in enumerate(range(260, 60, -40)):
        alpha = 6 + i * 3
        gdraw.ellipse(
            (WIDTH / 2 - r, 110 - r * 0.35, WIDTH / 2 + r, 110 + r * 0.35),
            fill=(79, 209, 197, alpha),
        )
    img = Image.alpha_composite(img.convert("RGBA"), glow).convert("RGB")
    draw = ImageDraw.Draw(img)

    # -- headline -----------------------------------------------------------
    f_common = font(FONT_BOLD, 80)
    f_adk = font(FONT_BOLD, 80)
    common_w, _ = text_size(draw, "Common", f_common)
    adk_w, _ = text_size(draw, "ADK", f_adk)
    total_w = common_w + adk_w
    start_x = WIDTH / 2 - total_w / 2
    headline_cy = 96

    l, t, r, b = draw.textbbox((0, 0), "Common", font=f_common)
    draw.text((start_x - l, headline_cy - (b - t) / 2 - t), "Common", font=f_common, fill=TEXT_PRIMARY)
    l2, t2, r2, b2 = draw.textbbox((0, 0), "ADK", font=f_adk)
    draw.text((start_x + common_w - l2, headline_cy - (b2 - t2) / 2 - t2), "ADK", font=f_adk, fill=TEAL)

    # -- subtitle -------------------------------------------------------------
    f_sub = font(FONT_REGULAR, 30)
    draw_centered(draw, WIDTH // 2, 168, "Define agents once. Build on any agent SDK.", f_sub, TEXT_MUTED)

    # -- motif: common/ source node fanning out to six SDK chips -------------
    motif_top, motif_bottom = 220, 560
    motif_cy = (motif_top + motif_bottom) // 2

    # source node -- rounded box with a small folder-tab, labeled "common/"
    node_w, node_h = 236, 110
    node_x0, node_y0 = 90, motif_cy - node_h // 2
    node_x1, node_y1 = node_x0 + node_w, node_y0 + node_h
    tab_w, tab_h = 70, 16
    rounded_rect(draw, (node_x0, node_y0 - tab_h, node_x0 + tab_w, node_y0 + 4), 6, fill=BOX_FILL, outline=WARM, width=2)
    rounded_rect(draw, (node_x0, node_y0, node_x1, node_y1), 14, fill=BOX_FILL, outline=WARM, width=3)
    f_node = font(FONT_BOLD, 30)
    draw_centered(draw, (node_x0 + node_x1) // 2, (node_y0 + node_y1) // 2, "common/", f_node, TEXT_PRIMARY)

    # six target chips, one column, evenly spaced -- each gets its own direct
    # line straight from the source node (a true fan-out, not a chain).
    chips = [
        "Google ADK",
        "OpenAI Agents",
        "Claude SDK",
        "CrewAI",
        "AutoGen",
        "LangGraph",
    ]
    chip_x0, chip_x1 = 760, 1190
    chip_w = chip_x1 - chip_x0
    chip_h = 50
    row_gap = 56
    first_cy = motif_cy - row_gap * 2.5
    row_cy = [first_cy + i * row_gap for i in range(6)]

    f_chip = font(FONT_BOLD, 23)
    node_right = (node_x1, motif_cy)

    chip_boxes = []
    for label, cy in zip(chips, row_cy):
        cy0 = cy - chip_h / 2
        cy1 = cy0 + chip_h
        chip_boxes.append((chip_x0, cy0, chip_x1, cy1, label, cy))

    # connector lines first (so chip fills sit on top of the line ends)
    for cx0, cy0, cx1, cy1, label, cy in chip_boxes:
        draw.line([node_right, (cx0, cy)], fill=LINE_COLOR, width=2)

    for cx0, cy0, cx1, cy1, label, cy in chip_boxes:
        rounded_rect(draw, (cx0, cy0, cx1, cy1), 10, fill=CHIP_FILL, outline=TEAL, width=2)
        tw, th = text_size(draw, label, f_chip)
        chip_font = f_chip
        if tw > chip_w - 24:
            chip_font = font(FONT_BOLD, 20)
        draw_centered(draw, (cx0 + cx1) / 2, cy, label, chip_font, TEXT_PRIMARY)

    # small accent dot at the node's fan-out point
    draw.ellipse((node_right[0] - 5, node_right[1] - 5, node_right[0] + 5, node_right[1] + 5), fill=WARM)

    # -- footer ---------------------------------------------------------------
    f_footer = font(FONT_REGULAR, 22)
    footer_text = "Apache-2.0 · Python · 6 SDK targets"
    draw.line([(WIDTH // 2 - 140, 592), (WIDTH // 2 + 140, 592)], fill=GRID, width=1)
    draw_centered(draw, WIDTH // 2, 616, footer_text, f_footer, TEXT_MUTED)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUT_PATH, "PNG")
    print(f"Wrote {OUT_PATH} ({img.width}x{img.height})")


if __name__ == "__main__":
    main()
