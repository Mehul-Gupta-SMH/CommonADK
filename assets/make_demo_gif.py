#!/usr/bin/env python3
"""Generate assets/demo.gif -- an animated terminal-mockup of commonadk running.

Run it directly (idempotent -- re-running regenerates byte-for-byte the same
GIF, since nothing here is randomized or timestamped):

    python3 assets/make_demo_gif.py

Every line of "terminal output" below is copied verbatim from a real run in
this environment (not fabricated):

    $ commonadk validate examples/research-crew/common
    $ python3 examples/demo.py

Both were actually executed while writing this script; see
docs/demo-runs.md for the full, unedited transcripts these lines are drawn
from. Long, already-documented sections (the demo script's env/render setup)
are elided with a visible "..." rather than dropped silently.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT_PATH = Path(__file__).resolve().parent / "demo.gif"

WIDTH, HEIGHT = 900, 560
TITLEBAR_H = 40
PAD_X, PAD_Y = 22, 16
FONT_SIZE = 14
LINE_H = 21
MAX_VISIBLE_LINES = (HEIGHT - TITLEBAR_H - PAD_Y * 2) // LINE_H

BG = "#0d1117"
CHROME = "#161b22"
CHROME_BORDER = "#30363d"
TEXT_PRIMARY = "#e6edf3"
TEXT_MUTED = "#8b949e"
PROMPT = "#f0a35c"
OK_COLOR = "#4fd1c5"

FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")
FONT_MONO = FONT_DIR / "DejaVuSansMono.ttf"
FONT_MONO_BOLD = FONT_DIR / "DejaVuSansMono-Bold.ttf"

# ---------------------------------------------------------------------------
# Terminal content -- every non-blank line here is real, captured output.
# (kind, text) where kind selects color/weight:
#   "prompt" -- a "$ ..." command line (warm accent, bold)
#   "out"    -- normal captured stdout (primary text)
#   "ok"     -- one of the six per-target "[target] OK -- ..." lines (teal)
#   "muted"  -- an elision marker or a section label
# ---------------------------------------------------------------------------
LINES: list[tuple[str, str]] = [
    ("prompt", "$ commonadk validate examples/research-crew/common"),
    ("out", "Project: research-crew  (entry agent: coordinator)"),
    ("out", ""),
    ("out", "Agents:"),
    ("out", "  coordinator"),
    ("out", "    model: fast -> gemini/gemini-2.5-flash"),
    ("out", "    tools: format_handoff_note, split_into_subtopics"),
    ("out", "    env: (none required)"),
    ("out", "  researcher"),
    ("out", "    model: gemini/gemini-2.5-pro -> gemini/gemini-2.5-pro"),
    ("out", "    tools: fetch_page, search_web"),
    ("out", "    env:"),
    ("out", "      TAVILY_API_KEY: not set (required) -- Search API key used by search_web"),
    ("out", "      POSTGRES_DSN: not set (optional) -- Connection string for citations db"),
    ("out", "  writer"),
    ("out", "    model: fast -> gemini/gemini-2.5-flash"),
    ("out", "    tools: count_words, format_as_markdown"),
    ("out", "    env: (none required)"),
    ("out", ""),
    ("prompt", "$ python3 examples/demo.py"),
    ("muted", "...  (sections 1-3: load/validate, render mermaid, satisfy env --"),
    ("muted", "     real output in docs/demo-runs.md)"),
    ("out", "4. Build 'coordinator' for all six supported targets"),
    ("out", "=============================================================="),
    ("ok", "[google-adk] OK -- google.adk.agents.llm_agent.LlmAgent"),
    ("ok", "[openai] OK -- agents.agent.Agent"),
    ("ok", "[claude] OK -- claude_agent_sdk.types.ClaudeAgentOptions"),
    ("ok", "[crewai] OK -- crewai.crew.Crew"),
    ("ok", "[autogen] OK -- autogen_agentchat.teams._group_chat._swarm_group_chat.Swarm"),
    ("ok", "[langgraph] OK -- langgraph.graph.state.CompiledStateGraph"),
    ("out", ""),
    ("muted", "...  (2 demonstrated failure modes -- see docs/demo-runs.md)"),
    ("out", ""),
    ("prompt", "$ echo $?"),
    ("out", "0"),
]

COLOR = {
    "prompt": PROMPT,
    "out": TEXT_PRIMARY,
    "ok": OK_COLOR,
    "muted": TEXT_MUTED,
}


def font_for(kind: str) -> ImageFont.FreeTypeFont:
    if kind == "prompt":
        return ImageFont.truetype(str(FONT_MONO_BOLD), FONT_SIZE)
    return ImageFont.truetype(str(FONT_MONO), FONT_SIZE)


def render_window(visible: list[tuple[str, str]], cursor: bool) -> Image.Image:
    img = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(img)

    # window chrome
    draw.rectangle((0, 0, WIDTH - 1, TITLEBAR_H), fill=CHROME)
    draw.line([(0, TITLEBAR_H), (WIDTH, TITLEBAR_H)], fill=CHROME_BORDER, width=1)
    draw.rectangle((0, 0, WIDTH - 1, HEIGHT - 1), outline=CHROME_BORDER, width=1)
    for i, color in enumerate(["#ff5f56", "#ffbd2e", "#27c93f"]):
        cx = 22 + i * 22
        cy = TITLEBAR_H // 2
        draw.ellipse((cx - 6, cy - 6, cx + 6, cy + 6), fill=color)
    title_font = ImageFont.truetype(str(FONT_MONO), 13)
    title = "commonadk -- research-crew"
    tb = draw.textbbox((0, 0), title, font=title_font)
    tw = tb[2] - tb[0]
    draw.text(((WIDTH - tw) / 2, TITLEBAR_H // 2 - 8), title, font=title_font, fill=TEXT_MUTED)

    # content
    y = TITLEBAR_H + PAD_Y
    for i, (kind, text) in enumerate(visible):
        f = font_for(kind)
        color = COLOR[kind]
        is_last = i == len(visible) - 1
        line_text = text
        if cursor and is_last:
            line_text = text + "▌"
        draw.text((PAD_X, y), line_text, font=f, fill=color)
        y += LINE_H

    return img


def build_frames() -> tuple[list[Image.Image], list[int]]:
    frames: list[Image.Image] = []
    durations: list[int] = []

    step = 3  # lines revealed per frame (within the 2-4 range)
    count = 0
    total = len(LINES)

    while count < total:
        count = min(count + step, total)
        window_lines = LINES[max(0, count - MAX_VISIBLE_LINES):count]
        frames.append(render_window(window_lines, cursor=(count % 6 < 3)))
        durations.append(450)

    # hold the final, fully-revealed frame for ~3s (a couple of blink frames
    # to keep the cursor animating, then a long static hold)
    final_lines = LINES[max(0, total - MAX_VISIBLE_LINES):total]
    frames.append(render_window(final_lines, cursor=True))
    durations.append(600)
    frames.append(render_window(final_lines, cursor=False))
    durations.append(2400)

    return frames, durations


def main() -> None:
    frames, durations = build_frames()
    # Palette-quantize for GIF size (256-color, dithered) while keeping crisp text.
    quantized = [f.convert("P", palette=Image.ADAPTIVE, colors=64) for f in frames]
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    quantized[0].save(
        OUT_PATH,
        save_all=True,
        append_images=quantized[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=2,
    )
    size_kb = OUT_PATH.stat().st_size / 1024
    print(f"Wrote {OUT_PATH} ({WIDTH}x{HEIGHT}, {len(frames)} frames, {size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
