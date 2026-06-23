from __future__ import annotations

import html
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont
from pygments import highlight
from pygments.formatters import ImageFormatter
from pygments.lexers import get_lexer_by_name
from pygments.util import ClassNotFound


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "ADR_Advanced_RAG_Architecture.md"
OUT_DIR = ROOT / "output" / "pdf"
PREVIEW_DIR = ROOT / "tmp" / "pdfs" / "preview"
PDF_PATH = OUT_DIR / "ADR_Advanced_RAG_Architecture.pdf"

PAGE_W = 1654
PAGE_H = 2339
MARGIN_X = 120
MARGIN_Y = 105
CONTENT_W = PAGE_W - 2 * MARGIN_X
BG = "white"
INK = "#1f2933"
MUTED = "#5f6b7a"
BLUE = "#1f5d99"
LIGHT_BLUE = "#eaf3ff"
GREEN = "#2d7a46"
LIGHT_GREEN = "#ecf8ef"
AMBER = "#9a6700"
LIGHT_AMBER = "#fff6df"
RED = "#a23b3b"
LIGHT_RED = "#fff1f1"
LINE = "#cfd7e3"
CODE_BG = "#f6f8fa"


def font_path(name: str) -> str:
    candidates = [
        Path("C:/Windows/Fonts") / name,
        Path("/usr/share/fonts/truetype/dejavu") / name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return name


FONT_REG = font_path("arial.ttf")
FONT_BOLD = font_path("arialbd.ttf")
FONT_MONO = font_path("consola.ttf")
FONT_MONO_BOLD = font_path("consolab.ttf")


def font(size: int, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont:
    if mono:
        return ImageFont.truetype(FONT_MONO_BOLD if bold else FONT_MONO, size)
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REG, size)


F_TITLE = font(44, True)
F_H2 = font(34, True)
F_H3 = font(27, True)
F_H4 = font(22, True)
F_BODY = font(22)
F_BODY_B = font(22, True)
F_SMALL = font(17)
F_CAPTION = font(18)
F_CODE = font(16, mono=True)
F_CODE_SMALL = font(14, mono=True)
F_NODE = font(17)
F_NODE_B = font(17, True)


@dataclass
class Block:
    kind: str
    text: str = ""
    level: int = 0
    lang: str = ""
    rows: list[list[str]] | None = None


def text_size(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.FreeTypeFont) -> tuple[int, int]:
    if not text:
        return 0, fnt.size
    box = draw.textbbox((0, 0), text, font=fnt)
    return box[2] - box[0], box[3] - box[1]


def wrap_text(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.FreeTypeFont, width: int) -> list[str]:
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return [""]
    words = text.split(" ")
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if text_size(draw, candidate, fnt)[0] <= width:
            current = candidate
        else:
            if current:
                lines.append(current)
            while text_size(draw, word, fnt)[0] > width and len(word) > 8:
                cut = max(8, int(width / max(1, text_size(draw, "m", fnt)[0])) - 2)
                lines.append(word[:cut] + "-")
                word = word[cut:]
            current = word
    if current:
        lines.append(current)
    return lines


def parse_markdown(md: str) -> list[Block]:
    lines = md.splitlines()
    blocks: list[Block] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            lang = line.strip("`").strip()
            i += 1
            body = []
            while i < len(lines) and not lines[i].startswith("```"):
                body.append(lines[i])
                i += 1
            i += 1
            blocks.append(Block("code", "\n".join(body), lang=lang))
            continue
        if not line.strip():
            i += 1
            continue
        if line.strip() == "---":
            blocks.append(Block("rule"))
            i += 1
            continue
        if line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            blocks.append(Block("heading", line[level:].strip(), level=level))
            i += 1
            continue
        if line.startswith("|") and i + 1 < len(lines) and lines[i + 1].startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].startswith("|"):
                if not re.match(r"^\|\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?$", lines[i]):
                    cells = [cell.strip() for cell in lines[i].strip().strip("|").split("|")]
                    table_lines.append(cells)
                i += 1
            blocks.append(Block("table", rows=table_lines))
            continue
        if line.lstrip().startswith("- "):
            items = []
            while i < len(lines) and lines[i].lstrip().startswith("- "):
                items.append(lines[i].lstrip()[2:].strip())
                i += 1
            blocks.append(Block("list", "\n".join(items)))
            continue
        if line.startswith(">"):
            quote = []
            while i < len(lines) and lines[i].startswith(">"):
                quote.append(lines[i].lstrip("> ").strip())
                i += 1
            blocks.append(Block("quote", " ".join(quote)))
            continue
        para = [line.strip()]
        i += 1
        while (
            i < len(lines)
            and lines[i].strip()
            and not lines[i].startswith("#")
            and not lines[i].startswith("```")
            and not lines[i].lstrip().startswith("- ")
            and not lines[i].startswith("|")
            and lines[i].strip() != "---"
        ):
            para.append(lines[i].strip())
            i += 1
        blocks.append(Block("para", " ".join(para)))
    return blocks


class PdfCanvas:
    def __init__(self) -> None:
        self.pages: list[Image.Image] = []
        self.page = Image.new("RGB", (PAGE_W, PAGE_H), BG)
        self.draw = ImageDraw.Draw(self.page)
        self.y = MARGIN_Y
        self.page_num = 1
        self.header_title = "ADR: Local-First Corrective RAG Architecture"
        self._draw_header()

    def _draw_header(self) -> None:
        self.draw.text((MARGIN_X, 45), self.header_title, font=F_SMALL, fill=MUTED)
        self.draw.line((MARGIN_X, 78, PAGE_W - MARGIN_X, 78), fill="#e5e9f0", width=2)

    def _draw_footer(self) -> None:
        y = PAGE_H - 55
        self.draw.line((MARGIN_X, y - 18, PAGE_W - MARGIN_X, y - 18), fill="#e5e9f0", width=2)
        self.draw.text((MARGIN_X, y), "Advanced RAG Architecture Decision Record", font=F_SMALL, fill=MUTED)
        page_label = f"Page {self.page_num}"
        w, _ = text_size(self.draw, page_label, F_SMALL)
        self.draw.text((PAGE_W - MARGIN_X - w, y), page_label, font=F_SMALL, fill=MUTED)

    def new_page(self) -> None:
        self._draw_footer()
        self.pages.append(self.page)
        self.page_num += 1
        self.page = Image.new("RGB", (PAGE_W, PAGE_H), BG)
        self.draw = ImageDraw.Draw(self.page)
        self.y = MARGIN_Y
        self._draw_header()

    def ensure(self, needed: int) -> None:
        if self.y + needed > PAGE_H - 105:
            self.new_page()

    def add_space(self, amount: int) -> None:
        self.y += amount

    def save(self) -> None:
        self._draw_footer()
        self.pages.append(self.page)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        self.pages[0].save(PDF_PATH, save_all=True, append_images=self.pages[1:], resolution=150)
        PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
        for idx, page in enumerate(self.pages[:8], start=1):
            preview = page.copy()
            preview.thumbnail((900, 1300))
            preview.save(PREVIEW_DIR / f"page-{idx:02d}.png")


def clean_inline(text: str) -> str:
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    return html.unescape(text)


def add_paragraph(c: PdfCanvas, text: str, fnt=F_BODY, fill=INK, indent=0, bullet: str | None = None) -> None:
    width = CONTENT_W - indent - (32 if bullet else 0)
    lines = wrap_text(c.draw, clean_inline(text), fnt, width)
    line_h = fnt.size + 9
    c.ensure(len(lines) * line_h + 16)
    x = MARGIN_X + indent
    if bullet:
        c.draw.ellipse((x + 3, c.y + 10, x + 13, c.y + 20), fill=BLUE)
        x += 32
    for line in lines:
        c.draw.text((x, c.y), line, font=fnt, fill=fill)
        c.y += line_h
    c.y += 8


def add_heading(c: PdfCanvas, block: Block) -> None:
    text = clean_inline(block.text)
    if block.level == 1:
        c.ensure(120)
        c.draw.rounded_rectangle((MARGIN_X, c.y, PAGE_W - MARGIN_X, c.y + 120), radius=18, fill="#17324d")
        c.draw.text((MARGIN_X + 30, c.y + 33), text, font=F_TITLE, fill="white")
        c.y += 155
    elif block.level == 2:
        c.ensure(75)
        c.y += 16
        c.draw.text((MARGIN_X, c.y), text, font=F_H2, fill="#17324d")
        c.y += 52
        c.draw.line((MARGIN_X, c.y, PAGE_W - MARGIN_X, c.y), fill="#bdd2e8", width=3)
        c.y += 24
    elif block.level == 3:
        c.ensure(55)
        c.y += 10
        c.draw.text((MARGIN_X, c.y), text, font=F_H3, fill=BLUE)
        c.y += 48
    else:
        c.ensure(42)
        c.draw.text((MARGIN_X, c.y), text, font=F_H4, fill=GREEN)
        c.y += 38


def add_rule(c: PdfCanvas) -> None:
    c.ensure(36)
    c.draw.line((MARGIN_X, c.y + 12, PAGE_W - MARGIN_X, c.y + 12), fill="#e5e9f0", width=2)
    c.y += 36


def add_quote(c: PdfCanvas, text: str) -> None:
    lines = wrap_text(c.draw, clean_inline(text), F_BODY, CONTENT_W - 60)
    h = len(lines) * 31 + 35
    c.ensure(h)
    c.draw.rounded_rectangle((MARGIN_X, c.y, PAGE_W - MARGIN_X, c.y + h), radius=12, fill="#f5f9ff", outline="#b9d3ee", width=2)
    c.draw.rectangle((MARGIN_X, c.y, MARGIN_X + 12, c.y + h), fill=BLUE)
    y = c.y + 18
    for line in lines:
        c.draw.text((MARGIN_X + 38, y), line, font=F_BODY, fill="#17324d")
        y += 31
    c.y += h + 20


def add_table(c: PdfCanvas, rows: list[list[str]]) -> None:
    if not rows:
        return
    cols = max(len(row) for row in rows)
    col_w = [CONTENT_W // cols] * cols
    row_heights = []
    wrapped: list[list[list[str]]] = []
    for r, row in enumerate(rows):
        wrapped_row = []
        max_lines = 1
        for i in range(cols):
            text = clean_inline(row[i] if i < len(row) else "")
            fnt = F_SMALL if r else font(17, True)
            lines = wrap_text(c.draw, text, fnt, col_w[i] - 22)
            wrapped_row.append(lines)
            max_lines = max(max_lines, len(lines))
        wrapped.append(wrapped_row)
        row_heights.append(max(44, max_lines * 23 + 18))
    total_h = sum(row_heights)
    if total_h > PAGE_H - 260:
        for row in rows:
            add_paragraph(c, " | ".join(row), F_SMALL, indent=18, bullet="-")
        return
    c.ensure(total_h + 18)
    y = c.y
    for r, row in enumerate(wrapped):
        x = MARGIN_X
        fill = "#eaf3ff" if r == 0 else ("#fbfdff" if r % 2 else "white")
        c.draw.rectangle((MARGIN_X, y, PAGE_W - MARGIN_X, y + row_heights[r]), fill=fill, outline=LINE)
        for i, lines in enumerate(row):
            c.draw.line((x, y, x, y + row_heights[r]), fill=LINE, width=1)
            fnt = font(17, True) if r == 0 else F_SMALL
            ty = y + 10
            for line in lines:
                c.draw.text((x + 11, ty), line, font=fnt, fill=INK)
                ty += 23
            x += col_w[i]
        c.draw.line((PAGE_W - MARGIN_X, y, PAGE_W - MARGIN_X, y + row_heights[r]), fill=LINE, width=1)
        y += row_heights[r]
    c.y = y + 24


def code_image(code: str, lang: str) -> Image.Image:
    lexer_name = {
        "python": "python",
        "bash": "bash",
        "json": "json",
        "text": "text",
    }.get(lang.lower(), "text")
    try:
        lexer = get_lexer_by_name(lexer_name)
    except ClassNotFound:
        lexer = get_lexer_by_name("text")
    formatter = ImageFormatter(
        font_name="Consolas",
        font_size=18,
        line_numbers=False,
        style="default",
        image_pad=18,
        line_pad=4,
        background_color=CODE_BG,
    )
    data = highlight(code.rstrip() or " ", lexer, formatter)
    img = Image.open(__import__("io").BytesIO(data)).convert("RGB")
    if img.width > CONTENT_W:
        ratio = CONTENT_W / img.width
        img = img.resize((CONTENT_W, max(1, int(img.height * ratio))), Image.LANCZOS)
    return img


def add_code(c: PdfCanvas, code: str, lang: str) -> None:
    if lang == "mermaid":
        add_mermaid(c, code)
        return
    img = code_image(code, lang)
    if img.height > PAGE_H - 260:
        lines = code.splitlines()
        chunk: list[str] = []
        for line in lines:
            chunk.append(line)
            if len(chunk) >= 34:
                add_code(c, "\n".join(chunk), lang)
                chunk = []
        if chunk:
            add_code(c, "\n".join(chunk), lang)
        return
    c.ensure(img.height + 44)
    c.draw.rounded_rectangle((MARGIN_X, c.y, MARGIN_X + img.width + 8, c.y + img.height + 8), radius=10, fill="#dde6ef")
    c.page.paste(img, (MARGIN_X + 4, c.y + 4))
    c.y += img.height + 32


def parse_node(token: str) -> tuple[str, str, str]:
    token = token.strip()
    m = re.match(r"([A-Za-z0-9_]+)\[\((.*?)\)\]", token)
    if m:
        return m.group(1), m.group(2), "round"
    m = re.match(r"([A-Za-z0-9_]+)\[\((.*?)\)\]", token)
    if m:
        return m.group(1), m.group(2), "round"
    m = re.match(r"([A-Za-z0-9_]+)\[\[(.*?)\]\]", token)
    if m:
        return m.group(1), m.group(2), "rect"
    m = re.match(r"([A-Za-z0-9_]+)\[\((.*?)\)\]", token)
    if m:
        return m.group(1), m.group(2), "round"
    m = re.match(r"([A-Za-z0-9_]+)\[\s*(.*?)\s*\]", token)
    if m:
        return m.group(1), m.group(2), "rect"
    m = re.match(r"([A-Za-z0-9_]+)\{\s*(.*?)\s*\}", token)
    if m:
        return m.group(1), m.group(2), "diamond"
    m = re.match(r"([A-Za-z0-9_]+)\(\[\s*(.*?)\s*\]\)", token)
    if m:
        return m.group(1), m.group(2), "round"
    m = re.match(r"([A-Za-z0-9_]+)\(\((.*?)\)\)", token)
    if m:
        return m.group(1), m.group(2), "round"
    m = re.match(r"([A-Za-z0-9_]+)\(\s*(.*?)\s*\)", token)
    if m:
        return m.group(1), m.group(2), "round"
    m = re.match(r"([A-Za-z0-9_]+)$", token)
    if m:
        return m.group(1), m.group(1), "rect"
    return token, token, "rect"


def parse_flowchart(src: str):
    lines = [line.strip() for line in src.splitlines() if line.strip()]
    direction = "TB"
    if lines and lines[0].startswith("flowchart"):
        parts = lines[0].split()
        if len(parts) > 1:
            direction = parts[1]
    nodes: dict[str, tuple[str, str]] = {}
    edges: list[tuple[str, str, str]] = []
    for line in lines[1:]:
        if line.startswith("subgraph") or line == "end":
            continue
        if "-->" not in line and "-.->" not in line and "==>" not in line:
            nid, label, shape = parse_node(line)
            nodes.setdefault(nid, (label, shape))
            continue

        arrow = "-->" if "-->" in line else ("-.->" if "-.->" in line else "==>")
        left_raw, right_raw = line.split(arrow, 1)
        left_raw = left_raw.strip()
        right_raw = right_raw.strip()
        label = ""

        label_m = re.match(r"^\|([^|]+)\|\s*(.+)$", right_raw)
        if label_m:
            label = label_m.group(1)
            right_raw = label_m.group(2).strip()
        elif " -- " in left_raw:
            source_raw, label_raw = left_raw.split(" -- ", 1)
            left_raw = source_raw.strip()
            label = label_raw.strip().strip('"')

        left_id, left_label, left_shape = parse_node(left_raw)
        right_id, right_label, right_shape = parse_node(right_raw)
        nodes.setdefault(left_id, (left_label, left_shape))
        nodes.setdefault(right_id, (right_label, right_shape))
        edges.append((left_id, right_id, label))
    return direction, nodes, edges


def layout_flow(direction: str, nodes: dict[str, tuple[str, str]], edges: list[tuple[str, str, str]]):
    indeg = {n: 0 for n in nodes}
    adj = {n: [] for n in nodes}
    for a, b, _ in edges:
        if a in nodes and b in nodes:
            adj[a].append(b)
            indeg[b] += 1
    level = {n: 0 for n in nodes}
    changed = True
    for _ in range(len(nodes) + 3):
        if not changed:
            break
        changed = False
        for a, b, _ in edges:
            if level.get(b, 0) <= level.get(a, 0):
                level[b] = level.get(a, 0) + 1
                changed = True
    groups: dict[int, list[str]] = {}
    for n, lvl in level.items():
        groups.setdefault(min(lvl, 5), []).append(n)
    ordered_levels = [groups[k] for k in sorted(groups)]
    return ordered_levels


def draw_arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], fill=BLUE) -> None:
    draw.line((start[0], start[1], end[0], end[1]), fill=fill, width=3)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    size = 12
    points = [
        end,
        (int(end[0] - size * math.cos(angle - 0.45)), int(end[1] - size * math.sin(angle - 0.45))),
        (int(end[0] - size * math.cos(angle + 0.45)), int(end[1] - size * math.sin(angle + 0.45))),
    ]
    draw.polygon(points, fill=fill)


def draw_node(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], label: str, shape: str) -> None:
    x1, y1, x2, y2 = box
    fill = LIGHT_BLUE if shape != "diamond" else LIGHT_AMBER
    outline = BLUE if shape != "diamond" else AMBER
    if shape == "diamond":
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        draw.polygon([(cx, y1), (x2, cy), (cx, y2), (x1, cy)], fill=fill, outline=outline)
    else:
        draw.rounded_rectangle(box, radius=16, fill=fill, outline=outline, width=3)
    max_w = x2 - x1 - 24
    lines = wrap_text(draw, label, F_NODE_B if shape == "diamond" else F_NODE, max_w)
    line_h = 22
    ty = y1 + (y2 - y1 - len(lines) * line_h) // 2
    for line in lines[:4]:
        w, _ = text_size(draw, line, F_NODE_B if shape == "diamond" else F_NODE)
        draw.text((x1 + (x2 - x1 - w) // 2, ty), line, font=F_NODE_B if shape == "diamond" else F_NODE, fill=INK)
        ty += line_h


def flowchart_image(src: str) -> Image.Image:
    direction, nodes, edges = parse_flowchart(src)
    levels = layout_flow(direction, nodes, edges)
    if not levels:
        levels = [list(nodes)]
    node_w, node_h = 220, 82
    gap_x, gap_y = 60, 76
    if direction == "LR":
        width = max(900, len(levels) * (node_w + gap_x) + 80)
        height = max(320, max(len(level) for level in levels) * (node_h + gap_y) + 80)
        positions = {}
        for li, level in enumerate(levels):
            total_h = len(level) * node_h + (len(level) - 1) * gap_y
            y0 = (height - total_h) // 2
            for ni, nid in enumerate(level):
                x = 45 + li * (node_w + gap_x)
                y = y0 + ni * (node_h + gap_y)
                positions[nid] = (x, y, x + node_w, y + node_h)
    else:
        width = max(900, max(len(level) for level in levels) * (node_w + gap_x) + 80)
        height = max(360, len(levels) * (node_h + gap_y) + 80)
        positions = {}
        for li, level in enumerate(levels):
            total_w = len(level) * node_w + (len(level) - 1) * gap_x
            x0 = (width - total_w) // 2
            for ni, nid in enumerate(level):
                x = x0 + ni * (node_w + gap_x)
                y = 45 + li * (node_h + gap_y)
                positions[nid] = (x, y, x + node_w, y + node_h)
    img = Image.new("RGB", (min(width, 1800), min(height, 1500)), "white")
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((5, 5, img.width - 6, img.height - 6), radius=20, fill="#fbfdff", outline="#dae3ee", width=2)
    for a, b, label in edges:
        if a not in positions or b not in positions:
            continue
        ax1, ay1, ax2, ay2 = positions[a]
        bx1, by1, bx2, by2 = positions[b]
        start = (ax2, (ay1 + ay2) // 2) if direction == "LR" else ((ax1 + ax2) // 2, ay2)
        end = (bx1, (by1 + by2) // 2) if direction == "LR" else ((bx1 + bx2) // 2, by1)
        draw_arrow(draw, start, end)
        if label:
            mx, my = (start[0] + end[0]) // 2, (start[1] + end[1]) // 2
            lw, lh = text_size(draw, label, F_SMALL)
            draw.rounded_rectangle((mx - lw // 2 - 8, my - 14, mx + lw // 2 + 8, my + 14), radius=8, fill="white", outline="#d5deea")
            draw.text((mx - lw // 2, my - lh // 2), label, font=F_SMALL, fill=MUTED)
    for nid, (label, shape) in nodes.items():
        if nid in positions:
            draw_node(draw, positions[nid], label, shape)
    return img


def sequence_image(src: str) -> Image.Image:
    lines = [line.strip() for line in src.splitlines() if line.strip()]
    participants: list[tuple[str, str]] = []
    messages: list[tuple[str, str, str]] = []
    aliases: dict[str, str] = {}
    for line in lines[1:]:
        pm = re.match(r"participant\s+(\w+)(?:\s+as\s+(.+))?", line)
        if pm:
            alias, label = pm.group(1), pm.group(2) or pm.group(1)
            aliases[alias] = label
            participants.append((alias, label))
            continue
        mm = re.match(r"(\w+)-+>>(\w+):\s*(.+)", line)
        if mm:
            messages.append((mm.group(1), mm.group(2), mm.group(3)))
    width = max(900, len(participants) * 210 + 100)
    height = max(360, len(messages) * 72 + 180)
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((5, 5, width - 6, height - 6), radius=20, fill="#fbfdff", outline="#dae3ee", width=2)
    x_positions = {}
    for i, (alias, label) in enumerate(participants):
        x = 80 + i * ((width - 160) // max(1, len(participants) - 1)) if len(participants) > 1 else width // 2
        x_positions[alias] = x
        draw.rounded_rectangle((x - 80, 35, x + 80, 85), radius=12, fill=LIGHT_BLUE, outline=BLUE, width=2)
        display = label[:24]
        w, _ = text_size(draw, display, F_NODE_B)
        draw.text((x - w // 2, 51), display, font=F_NODE_B, fill=INK)
        draw.line((x, 85, x, height - 45), fill="#b8c4d3", width=2)
    y = 130
    for a, b, label in messages:
        if a not in x_positions or b not in x_positions:
            continue
        x1, x2 = x_positions[a], x_positions[b]
        draw_arrow(draw, (x1, y), (x2, y))
        lines_wrapped = wrap_text(draw, label, F_SMALL, abs(x2 - x1) + 110)
        tx = min(x1, x2) + 15
        for line in lines_wrapped[:2]:
            draw.text((tx, y - 28), line, font=F_SMALL, fill=INK)
            y += 18
        y += 54
    return img


def add_mermaid(c: PdfCanvas, src: str) -> None:
    if src.strip().startswith("sequenceDiagram"):
        img = sequence_image(src)
    elif src.strip().startswith("flowchart"):
        img = flowchart_image(src)
    else:
        img = code_image(src, "text")
    if img.width > CONTENT_W:
        ratio = CONTENT_W / img.width
        img = img.resize((CONTENT_W, max(1, int(img.height * ratio))), Image.LANCZOS)
    c.ensure(img.height + 58)
    c.draw.text((MARGIN_X, c.y), "Rendered Mermaid diagram", font=F_CAPTION, fill=MUTED)
    c.y += 28
    c.draw.rounded_rectangle((MARGIN_X, c.y, MARGIN_X + img.width + 8, c.y + img.height + 8), radius=14, fill="#dae3ee")
    c.page.paste(img, (MARGIN_X + 4, c.y + 4))
    c.y += img.height + 34


def render(blocks: Iterable[Block]) -> None:
    c = PdfCanvas()
    for block in blocks:
        if block.kind == "heading":
            add_heading(c, block)
        elif block.kind == "para":
            add_paragraph(c, block.text)
        elif block.kind == "list":
            for item in block.text.splitlines():
                add_paragraph(c, item, bullet="-")
        elif block.kind == "rule":
            add_rule(c)
        elif block.kind == "quote":
            add_quote(c, block.text)
        elif block.kind == "table":
            add_table(c, block.rows or [])
        elif block.kind == "code":
            add_code(c, block.text, block.lang)
    c.save()


def main() -> None:
    md = SOURCE.read_text(encoding="utf-8")
    blocks = parse_markdown(md)
    render(blocks)
    print(PDF_PATH)
    print(PREVIEW_DIR)


if __name__ == "__main__":
    main()
